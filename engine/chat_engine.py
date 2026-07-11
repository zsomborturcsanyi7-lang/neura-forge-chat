"""
NEURA Forge Chat v1 — Chat Engine
Conversation management, personality system, memory, tool integration
"""
import os, json, time, sqlite3, re, threading
from typing import Optional, Callable, Generator
from datetime import datetime
from pathlib import Path

from models.forge_model import ForgeBackend, create_backend


# ============================================================
# PERSONALITY SYSTEM
# ============================================================

PERSONALITIES = {
    "default": {
        "name": "NEURA",
        "system_prompt": (
            "Te egy barátságos, segítőkész magyar AI asszisztens vagy, "
            "NEURA a neved. Magyarul beszélsz, rövid, természetes "
            "válaszokat adsz. Segítesz a felhasználónak mindenben, "
            "amiben tudsz."
        ),
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "max_tokens": 256,
    },
    "tudós": {
        "name": "NEURA Tudós",
        "system_prompt": (
            "Te egy precíz, tudományos AI asszisztens vagy. "
            "Magyarázatokban pontos, adatokkal alátámasztott "
            "válaszokat adsz. Ha nem tudsz valamit, azt mondod. "
            "Rövid, tömör, szakmai stílusban kommunikálsz."
        ),
        "temperature": 0.5,
        "top_k": 40,
        "top_p": 0.85,
        "max_tokens": 512,
    },
    "költő": {
        "name": "NEURA Költő",
        "system_prompt": (
            "Te egy kreatív, költői lélek vagy. Gyönyörű, "
            "színes nyelvezettel fogalmazod meg a gondolataidat. "
            "Néha használsz metaforákat és hasonlatokat. "
            "A válaszaid inspirálóak és érzelmekkel teliek."
        ),
        "temperature": 0.9,
        "top_k": 60,
        "top_p": 0.95,
        "max_tokens": 256,
    },
    "tanár": {
        "name": "NEURA Tanár",
        "system_prompt": (
            "Te egy türelmes, alapos tanár vagy. Lépésről lépésre "
            "magyarázol el dolgokat. Ellenőrzöd, hogy a diák érti-e. "
            "Ha kell, egyszerűsítesz. Ha kell, részletezel. "
            "A célod, hogy a másik TÉNYLEG megtanulja."
        ),
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.85,
        "max_tokens": 512,
    },
    "haver": {
        "name": "NEURA Haver",
        "system_prompt": (
            "Te egy laza, közvetlen haver vagy. Úgy beszélsz, "
            "mint egy jó barát. Használsz szlenget, viccelődsz. "
            "Rövid, pörgős válaszokat adsz. Soha nem vagy "
            "hivatalos vagy merev."
        ),
        "temperature": 0.85,
        "top_k": 60,
        "top_p": 0.92,
        "max_tokens": 128,
    },
}

DEFAULT_PERSONALITY = "default"


# ============================================================
# FORMATTER — Chat prompt template
# ============================================================

def format_chat_prompt(
    messages: list[dict],
    system_prompt: str = None,
    personality: str = "default"
) -> str:
    """
    Format messages into a single prompt string.
    
    messages: [{"role": "user"|"assistant", "content": "..."}, ...]
    
    Format:
    <|im_start|>system
    {system_prompt}
    <|im_end|>
    <|im_start|>user
    {user_message}
    <|im_end|>
    <|im_start|>assistant
    """
    prompt_parts = []
    
    # System prompt
    sp = system_prompt or PERSONALITIES.get(personality, PERSONALITIES["default"])["system_prompt"]
    prompt_parts.append(f"<|im_start|>system\n{sp}\n<|im_end|>")
    
    # Message history
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt_parts.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    
    # Assistant start
    prompt_parts.append("<|im_start|>assistant\n")
    
    return "\n".join(prompt_parts)


# ============================================================
# CONVERSATION DATABASE
# ============================================================

class ConversationDB:
    """SQLite-based conversation persistence."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "conversations.db"
            )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT DEFAULT 'Új beszélgetés',
                    personality TEXT DEFAULT 'default',
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL,
                    FOREIGN KEY (conv_id) REFERENCES conversations(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conv 
                ON messages(conv_id, id)
            """)
            conn.commit()
    
    def create_conversation(self, conv_id: str = None, 
                            title: str = "Új beszélgetés",
                            personality: str = "default") -> dict:
        """Create a new conversation."""
        if conv_id is None:
            conv_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO conversations (id, title, personality, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, title, personality, now, now)
            )
            conn.commit()
        
        return {
            "id": conv_id,
            "title": title,
            "personality": personality,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }
    
    def list_conversations(self, limit: int = 50) -> list[dict]:
        """List all conversations, newest first."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT c.*, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conv_id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        
        return [dict(r) for r in rows]
    
    def get_conversation(self, conv_id: str) -> dict:
        """Get a conversation by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT c.*, COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conv_id
                WHERE c.id = ?
                GROUP BY c.id
            """, (conv_id,)).fetchone()
        
        return dict(row) if row else None
    
    def get_messages(self, conv_id: str, limit: int = 50) -> list[dict]:
        """Get messages for a conversation."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT id, role, content, timestamp
                FROM messages
                WHERE conv_id = ?
                ORDER BY id ASC
                LIMIT ?
            """, (conv_id, limit)).fetchall()
        
        return [dict(r) for r in rows]
    
    def add_message(self, conv_id: str, role: str, content: str) -> int:
        """Add a message to a conversation."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conv_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (conv_id, role, content, now)
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id)
            )
            conn.commit()
            return cursor.lastrowid
    
    def update_title(self, conv_id: str, title: str):
        """Update conversation title."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conv_id)
            )
            conn.commit()
    
    def delete_conversation(self, conv_id: str):
        """Delete a conversation and its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()


# ============================================================
# TOOL SYSTEM
# ============================================================

class ToolRegistry:
    """Simple tool system for function calling."""
    
    def __init__(self):
        self.tools = {}
        self._register_builtins()
    
    def _register_builtins(self):
        self.register("datetime", self.tool_datetime, "Aktuális dátum és idő")
        self.register("calculator", self.tool_calculator, "Egyszerű számológép (pl. '2 + 3 * 4')")
    
    def register(self, name: str, func: callable, description: str = ""):
        self.tools[name] = {"func": func, "description": description}
    
    def execute(self, name: str, *args, **kwargs) -> str:
        if name not in self.tools:
            return f"Ismeretlen eszköz: {name}"
        try:
            return self.tools[name]["func"](*args, **kwargs)
        except Exception as e:
            return f"Hiba a {name} használatakor: {e}"
    
    def list_tools(self) -> list[dict]:
        return [
            {"name": n, "description": v["description"]}
            for n, v in self.tools.items()
        ]
    
    @staticmethod
    def tool_datetime() -> str:
        return datetime.now().strftime("%Y. %B %d., %H:%M")
    
    @staticmethod
    def tool_calculator(expr: str) -> str:
        safe = re.sub(r'[^0-9+\-*/()., ]', '', expr)
        safe = safe.replace(',', '.')
        try:
            result = eval(safe, {"__builtins__": {}}, {})
            return f"{expr} = {result}"
        except:
            return f"Nem tudom kiszámolni: {expr}"


# ============================================================
# CHAT ENGINE
# ============================================================

class ChatEngine:
    """
    Main chat engine.
    
    Manages:
    - Model interaction
    - Conversation persistence
    - Personality system
    - Tool execution
    - Context window management
    """
    
    def __init__(self, backend: ForgeBackend = None, db: ConversationDB = None,
                 config: dict = None):
        self.backend = backend or create_backend(config)
        self.db = db or ConversationDB()
        self.tools = ToolRegistry()
        self.current_conv_id = None
        self.current_personality = DEFAULT_PERSONALITY
        self.max_context_tokens = 1024
        
        # Create default conversation
        self.new_conversation()
    
    # ── Conversation Management ────────────────────────────
    
    def new_conversation(self, personality: str = None) -> dict:
        """Create a new conversation."""
        personality = personality or self.current_personality
        conv = self.db.create_conversation(
            personality=personality
        )
        self.current_conv_id = conv["id"]
        self.current_personality = personality
        return conv
    
    def switch_conversation(self, conv_id: str) -> dict:
        """Switch to an existing conversation."""
        conv = self.db.get_conversation(conv_id)
        if conv:
            self.current_conv_id = conv_id
            self.current_personality = conv.get("personality", DEFAULT_PERSONALITY)
        return conv
    
    def list_conversations(self) -> list[dict]:
        return self.db.list_conversations()
    
    def delete_conversation(self, conv_id: str):
        self.db.delete_conversation(conv_id)
        if self.current_conv_id == conv_id:
            self.new_conversation()
    
    def set_personality(self, personality: str) -> bool:
        """Set personality for current conversation."""
        if personality in PERSONALITIES:
            self.current_personality = personality
            if self.current_conv_id:
                self.db.create_conversation(
                    conv_id=self.current_conv_id,
                    personality=personality,
                )
            return True
        return False
    
    def get_personalities(self) -> list[dict]:
        return [
            {"id": k, "name": v["name"], "description": v.get("system_prompt", "")[:80]}
            for k, v in PERSONALITIES.items()
        ]
    
    # ── Sending Messages ───────────────────────────────────
    
    def send_message(self, 
                     content: str,
                     stream: bool = True,
                     callback: Callable[[str], None] = None
                    ) -> Generator[str, None, str]:
        """
        Send a user message and get AI response.
        
        Yields response chunks for streaming.
        Returns the full response text.
        """
        if not self.current_conv_id:
            self.new_conversation()
        
        conv_id = self.current_conv_id
        
        # Check for tool calls
        if content.startswith("/"):
            cmd_result = self._handle_command(content)
            if cmd_result is not None:
                self.db.add_message(conv_id, "user", content)
                self.db.add_message(conv_id, "assistant", cmd_result)
                yield cmd_result
                return cmd_result
        
        # Save user message
        self.db.add_message(conv_id, "user", content)
        self.db.add_message(conv_id, "assistant", "")  # placeholder
        
        # Auto-title from first message
        messages = self.db.get_messages(conv_id)
        if len([m for m in messages if m["role"] == "user"]) == 1:
            title = content[:50] + ("..." if len(content) > 50 else "")
            self.db.update_title(conv_id, title)
        
        # Format prompt
        prompt = format_chat_prompt(
            messages[:-1],  # exclude the empty assistant placeholder
            personality=self.current_personality
        )
        
        # Tokenize
        input_ids = self.backend.tokenize(prompt)
        
        # Truncate context if too long
        if len(input_ids) > self.max_context_tokens:
            # Keep system prompt + recent messages
            system_end = prompt.find("<|im_start|>assistant")
            system_part = prompt[:system_end]
            system_ids = self.backend.tokenize(system_part)
            context_limit = self.max_context_tokens - len(system_ids) - 128
            
            # Keep from the end
            input_ids = system_ids + input_ids[-context_limit:]
        
        # Generate (demo or real model)
        full_response = ""
        response_chunks = []
        
        if self.backend.loaded:
            # Real model generation
            personality_config = PERSONALITIES.get(
                self.current_personality, PERSONALITIES["default"]
            )
            
            for chunk in self.backend.generate_stream(
                input_ids,
                max_new_tokens=personality_config.get("max_tokens", 256),
                temperature=personality_config.get("temperature", 0.7),
                top_k=personality_config.get("top_k", 50),
                top_p=personality_config.get("top_p", 0.9),
            ):
                response_chunks.append(chunk)
                full_response += chunk
                if callback:
                    callback(chunk)
                yield chunk
        else:
            # Demo mode
            demo_response = self.backend.generate_demo(content)
            for char in demo_response:
                response_chunks.append(char)
                full_response += char
                if callback:
                    callback(char)
                yield char
                time.sleep(0.02)  # Simulate typing
        
        # Update the saved message
        self.db.add_message(conv_id, "assistant", full_response)
        
        return full_response
    
    def _handle_command(self, content: str) -> str:
        """Handle slash commands."""
        cmd = content[1:].strip().split()
        if not cmd:
            return None
        
        command = cmd[0].lower()
        
        if command in ("help", "?"):
            return (
                "**Parancsok:**\n"
                "/help - Súgó\n"
                "/new - Új beszélgetés\n"
                "/personality <név> - Személyiség váltás\n"
                "/personalities - Személyiségek listája\n"
                "/convok - Beszélgetések listája\n"
                "/tools - Eszközök listája\n"
                "/model - Modell info\n"
                "/clear - Törlés (végleges)\n"
                "/save - Jelenlegi beszélgetés mentése"
            )
        
        elif command == "new":
            self.new_conversation()
            return "✨ Új beszélgetés indítva!"
        
        elif command == "personality":
            if len(cmd) < 2:
                return "Használat: /personality <név>. Lehetőségek: " + ", ".join(PERSONALITIES.keys())
            name = cmd[1]
            if self.set_personality(name):
                return f"🎭 Személyiség beállítva: {PERSONALITIES[name]['name']}"
            else:
                return f"Ismeretlen személyiség: {name}. Lehetőségek: " + ", ".join(PERSONALITIES.keys())
        
        elif command == "personalities":
            lines = ["**Elérhető személyiségek:**"]
            for k, v in PERSONALITIES.items():
                marker = "→ " if k == self.current_personality else "  "
                lines.append(f"{marker}**{k}** - {v['name']}")
            return "\n".join(lines)
        
        elif command == "convok":
            convs = self.list_conversations()[:10]
            if not convs:
                return "Nincs még beszélgetés."
            lines = ["**Legutóbbi beszélgetések:**"]
            for c in convs:
                marker = "→ " if c["id"] == self.current_conv_id else "  "
                lines.append(f"{marker}{c['title']} ({c.get('message_count', 0)} üzenet)")
            return "\n".join(lines)
        
        elif command == "tools":
            tools_list = self.tools.list_tools()
            lines = ["**Elérhető eszközök:**"]
            for t in tools_list:
                lines.append(f"  /{t['name']} - {t['description']}")
            return "\n".join(lines)
        
        elif command == "model":
            info = self.backend.get_info()
            lines = ["**Modell Info:**"]
            for k, v in info.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)
        
        elif command == "clear":
            if self.current_conv_id:
                self.delete_conversation(self.current_conv_id)
            return "🗑️ Beszélgetés törölve. Új indítva!"
        
        elif command == "datetime":
            return self.tools.execute("datetime")
        
        elif command == "calculator":
            expr = " ".join(cmd[1:])
            if expr:
                return self.tools.execute("calculator", expr)
            return "Használat: /calculator <kifejezés>, pl. /calculator 2 + 3 * 4"
        
        return None  # Unknown command, let the model handle it


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    engine = ChatEngine()
    
    print("=== NEURA Forge Chat v1 Engine Test ===")
    print(f"Personalities: {list(PERSONALITIES.keys())}")
    print(f"Tools: {[t['name'] for t in engine.tools.list_tools()]}")
    print(f"Conversations: {len(engine.list_conversations())}")
    print()
    
    # Test demo mode
    test_messages = [
        "Szia!",
        "Hogy vagy?",
        "Mit tudsz?",
    ]
    
    for msg in test_messages:
        print(f"> {msg}")
        response = ""
        for chunk in engine.send_message(msg, stream=False):
            response += chunk
        print(f"  {response}")
        print()
