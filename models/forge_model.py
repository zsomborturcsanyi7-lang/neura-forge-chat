"""
NEURA Forge Chat v1 — Model Backend
Custom architecture matching the pulse_350m pretrained checkpoint
"""
import os, sys, time, json, math, re
from typing import Optional, Callable, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm


# ============================================================
# CONFIG
# ============================================================
DEFAULT_CONFIG = {
    "model_path": os.path.expanduser(
        "~/Desktop/MicroLanguageSwarm/pulse_350m_pretrained.pt"
    ),
    "assistant_path": os.path.expanduser(
        "~/Desktop/MicroLanguageSwarm/pulse_350m_assistant_checkpoint.pt"
    ),
    "tokenizer_paths": [
        os.path.expanduser("~/Desktop/MicroLanguageSwarm/data/bitnet_pretrain/tokenizer/tokenizer.model"),
        os.path.expanduser("~/Desktop/MicroLanguageSwarm/bitnet_kaggle_data/tokenizer.model"),
    ],
    "device": "cpu",
    "max_seq_len": 2048,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
}


# ============================================================
# ARCHITECTURE — Matches checkpoint state dict exactly
# ============================================================

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class RMSNorm(nn.Module):
    """RMS Normalization (matches 'norm1', 'norm2', 'norm' keys)."""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class FusedQKVAttention(nn.Module):
    """
    Fused QKV multi-head attention.
    
    Checkpoint has: attn.qkv.weight [3072, 1024]
    where 3072 = 3 * 1024 (fused Q, K, V for all heads)
    """
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Fused QKV: 3 * d_model -> all heads combined
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        
        # RoPE frequencies
        self.register_buffer("freqs_cis", 
            precompute_freqs_cis(self.head_dim, max_seq_len))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device
        
        # Fused QKV
        qkv = self.qkv(x)  # [B, T, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, T, d_model]
        
        # Reshape to [B, T, n_heads, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        
        # RoPE
        freqs = self.freqs_cis[:T].unsqueeze(0).unsqueeze(2)  # [1, T, 1, head_dim/2]
        
        q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        q_out = torch.view_as_real(q_complex * freqs).reshape(B, T, self.n_heads, self.head_dim)
        k_out = torch.view_as_real(k_complex * freqs).reshape(B, T, self.n_heads, self.head_dim)
        
        q = q_out.type_as(q)
        k = k_out.type_as(k)
        
        # Flash attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o(out)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN (matches 'mlp.w1', 'mlp.w2', 'mlp.w3' keys).
    
    Checkpoint has:
      mlp.w1.weight: [2730, 1024]  — gate
      mlp.w2.weight: [2730, 1024]  — up
      mlp.w3.weight: [1024, 2730]  — down
    """
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.w2 = nn.Linear(d_model, hidden, bias=False)  # up
        self.w3 = nn.Linear(hidden, d_model, bias=False)  # down
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class ForgeBlock(nn.Module):
    """Single transformer block matching checkpoint keys."""
    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, max_seq_len: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = FusedQKVAttention(d_model, n_heads, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLUFFN(d_model, ffn_hidden)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ForgeModel(nn.Module):
    """
    Pulse 350M — Hungarian language model.
    
    Architecture matching pulse_350m_pretrained.pt checkpoint:
    - 24 layers, 1024 dim, 16 heads, 2730 FFN hidden
    - Fused QKV attention + RoPE + SwiGLU
    - RMSNorm (not SubLN)
    """
    def __init__(self, vocab_size: int = 32000, d_model: int = 1024,
                 n_heads: int = 16, n_layers: int = 24, 
                 ffn_hidden: int = None, max_seq_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.embed = nn.Embedding(vocab_size, d_model)
        
        if ffn_hidden is None:
            ffn_hidden = int(d_model * 8 / 3)  # 2730
        
        self.layers = nn.ModuleList([
            ForgeBlock(d_model, n_heads, ffn_hidden, max_seq_len)
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 0.7, top_k: int = 50, 
                 top_p: float = 0.9, eos_token_id: int = 3) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        device = input_ids.device
        batch_size, seq_len = input_ids.shape
        
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.max_seq_len:
                inp = input_ids[:, -self.max_seq_len:]
            else:
                inp = input_ids
            
            logits = self(inp)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k
            if top_k > 0:
                vals, _ = torch.topk(next_logits, top_k, dim=-1)
                threshold = vals[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float('-inf')
            
            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs > top_p
                mask[:, 1:] = mask[:, :-1].clone()
                mask[:, 0] = False
                sorted_logits[mask] = float('-inf')
                next_logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
        
        return input_ids[:, seq_len:]


# ============================================================
# FORGE BACKEND — Model Wrapper with tokenizer + streaming
# ============================================================

class ForgeBackend:
    """
    High-level model wrapper.
    Handles loading, tokenization, generation, streaming.
    """
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = None
        self.tokenizer = None
        self.device = torch.device(self.config["device"])
        self.loaded = False
        self.demo_mode = config.get("demo_mode", False) if config else False
        self.stats = {}
    
    # ── Tokenizer ──────────────────────────────────────────
    
    def load_tokenizer(self) -> bool:
        """Load SentencePiece tokenizer."""
        sp = spm.SentencePieceProcessor()
        
        for path in self.config.get("tokenizer_paths", []):
            if os.path.exists(path):
                sp.Load(path)
                print(f"  [✓] Tokenizer: {os.path.basename(path)} (vocab={sp.GetPieceSize()})")
                self.tokenizer = sp
                self.stats["vocab_size"] = sp.GetPieceSize()
                return True
        
        # Fallback: search
        for root in [os.path.expanduser("~/Desktop/MicroLanguageSwarm"), 
                     os.path.expanduser("~/Desktop")]:
            if os.path.exists(root):
                for dirpath, _, files in os.walk(root):
                    for f in files:
                        if f == "tokenizer.model":
                            full = os.path.join(dirpath, f)
                            try:
                                sp.Load(full)
                                print(f"  [✓] Tokenizer (found): {full} (vocab={sp.GetPieceSize()})")
                                self.tokenizer = sp
                                self.stats["vocab_size"] = sp.GetPieceSize()
                                return True
                            except:
                                continue
    
        print("  [!] No tokenizer found!")
        return False
    
    def tokenize(self, text: str) -> list[int]:
        if self.tokenizer is None:
            return []
        if hasattr(self.tokenizer, 'EncodeAsIds'):
            return self.tokenizer.EncodeAsIds(text)
        return self.tokenizer.Encode(text, out_type=int)
    
    def detokenize(self, ids: list[int]) -> str:
        if self.tokenizer is None:
            return ""
        if hasattr(self.tokenizer, 'DecodeIds'):
            return self.tokenizer.DecodeIds(ids)
        return self.tokenizer.Decode(ids)
    
    # ── Model Loading ──────────────────────────────────────
    
    def load_model(self, checkpoint_path: str = None, 
                   use_assistant: bool = False) -> bool:
        """Load model from checkpoint."""
        if use_assistant:
            ckpt_path = checkpoint_path or self.config["assistant_path"]
        else:
            ckpt_path = checkpoint_path or self.config["model_path"]
        
        if not os.path.exists(ckpt_path):
            print(f"  [!] Checkpoint not found: {ckpt_path}")
            return False
        
        size_gb = os.path.getsize(ckpt_path) / 1024**3
        print(f"  [i] Loading: {os.path.basename(ckpt_path)} ({size_gb:.1f} GB)")
        
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        
        # Get architecture params
        vocab_size = ckpt.get('vocab_size', 32000)
        d_model = ckpt.get('d_model', 1024)
        n_heads = ckpt.get('n_heads', 16)
        n_layers = ckpt.get('n_layers', 24)
        
        # Extract state dict
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        else:
            # Try to filter out non-weight keys
            state_dict = {k: v for k, v in ckpt.items() 
                         if hasattr(v, 'shape') and ('weight' in k or 'bias' in k or 'freqs' in k)}
        
        # Build model
        model = ForgeModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=self.config["max_seq_len"]
        )
        
        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing and len(missing) < 10:
            print(f"  [!] Missing: {missing}")
        elif missing:
            # Try to remap keys
            remap = {}
            for k in missing:
                # Try common renames
                alt = k.replace('forge_block.', 'mlp.')  # no, keep as is
                if k in state_dict:
                    remap[k] = state_dict[k]
            
            if remap:
                model.load_state_dict(remap, strict=False)
                print(f"  [i] Remapped {len(remap)} keys")
        
        if unexpected:
            print(f"  [!] Unexpected keys: {len(unexpected)} (may include optimizer state)")
        
        model.eval()
        model = model.to(self.device)
        
        self.model = model
        self.loaded = True
        self.stats.update({
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "checkpoint": os.path.basename(ckpt_path),
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        })
        
        print(f"  [✓] Model loaded: {self.stats['params_m']:.1f}M params")
        return True
    
    def load_assistant(self) -> bool:
        return self.load_model(self.config["assistant_path"], use_assistant=True)
    
    # ── Generation ─────────────────────────────────────────
    
    @torch.no_grad()
    def generate_stream(self, input_ids: list[int], max_new_tokens: int = 128,
                        temperature: float = None, top_k: int = None,
                        top_p: float = None, stop_tokens: list[int] = None,
                        callback: Callable[[str], None] = None) -> Generator[str, None, None]:
        """Streaming generation."""
        if not self.loaded:
            yield "⚠️ Modell nincs betöltve."
            return
        
        temp = temperature if temperature is not None else self.config["temperature"]
        tk = top_k if top_k is not None else self.config["top_k"]
        tp = top_p if top_p is not None else self.config["top_p"]
        stop = stop_tokens or [0, 2, 3]
        
        x = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        for step in range(max_new_tokens):
            if x.shape[1] > self.config["max_seq_len"]:
                inp = x[:, -self.config["max_seq_len"]:]
            else:
                inp = x
            
            logits = self.model(inp)
            next_logits = logits[0, -1, :] / temp
            
            # Top-k
            if tk > 0:
                vals, _ = torch.topk(next_logits, min(tk, next_logits.shape[-1]))
                next_logits[next_logits < vals[-1]] = float('-inf')
            
            # Top-p
            if tp < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cum_probs > tp
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_logits[mask] = float('-inf')
                next_logits = torch.zeros_like(next_logits).scatter_(
                    0, sorted_indices, sorted_logits)
            
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            
            if next_id in stop:
                break
            
            chunk = self.detokenize([next_id])
            if callback:
                callback(chunk)
            yield chunk
            
            x = torch.cat([x, torch.tensor([[next_id]], device=self.device)], dim=1)
    
    def generate(self, input_ids: list[int], **kwargs) -> str:
        """Non-streaming generation."""
        chunks = []
        for chunk in self.generate_stream(input_ids, **kwargs):
            chunks.append(chunk)
        return "".join(chunks)
    
    # ── Demo Mode ──────────────────────────────────────────
    
    def generate_demo(self, prompt: str) -> str:
        """Demo mode responses."""
        p = prompt.lower()
        if any(g in p for g in ["szia", "hello", "sziasztok", "hali"]):
            return "Szia! Örülök, hogy beszélgetsz velem. Hogy segíthetek?"
        if "hogy vagy" in p:
            return "Köszönöm, jól vagyok! És te hogy érzed magad?"
        if "mit tudsz" in p:
            return ("Sok mindenre képes vagyok! Tudok beszélgetni, segíteni "
                   "a tanulásban, ötleteket adni, vagy csak társalogni. "
                   "Bár még fiatal vagyok, igyekszem egyre okosabb lenni!")
        if "nev" in p:
            return "A nevem NEURA! Azért vagyok itt, hogy segítsek neked."
        if "kösz" in p or "koszi" in p:
            return "Szívesen! Bármikor számíthatsz rám."
        return ("Ez egy érdekes téma! Sajnos még nem tudok mindenre válaszolni, "
                "de szívesen beszélgetek róla. Mesélj bővebben!")
    
    # ── Info ───────────────────────────────────────────────
    
    def get_info(self) -> dict:
        info = {
            "loaded": self.loaded,
            "demo": self.demo_mode,
            "device": str(self.device),
        }
        info.update(self.stats)
        return info


# ============================================================
# Quick init
# ============================================================
def create_backend(config: dict = None) -> ForgeBackend:
    """Create and load backend."""
    backend = ForgeBackend(config)
    
    print("╔══════════════════════════════════════════╗")
    print("║   NEURA Forge Chat v1 — Model Loader    ║")
    print("╚══════════════════════════════════════════╝")
    print()
    
    # Tokenizer
    if not backend.load_tokenizer():
        print("  [!] Demo mode (no tokenizer)")
        backend.demo_mode = True
        return backend
    
    # Try to load model
    try:
        if not backend.load_model():
            print("  [i] Trying assistant checkpoint...")
            if not backend.load_assistant():
                print("  [!] Demo mode (no model checkpoint)")
                backend.demo_mode = True
    except Exception as e:
        print(f"  [!] Model load error: {e}")
        backend.demo_mode = True
    
    print()
    return backend


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    backend = create_backend()
    
    print("=== TEST ===")
    if backend.loaded:
        result = backend.generate(
            backend.tokenize("Szia! Hogy vagy?"),
            max_new_tokens=30
        )
        print(f"  Output: '{result}'")
    
    print(f"\nInfo: {json.dumps(backend.get_info(), indent=2)}")
