"""
NEURA Forge Chat v1 — CLI Client
Interactive chat in the terminal
"""
import os, sys, time, threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.chat_engine import ChatEngine, PERSONALITIES


# ============================================================
# ANSI COLORS
# ============================================================

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    
    PURPLE = "\033[38;5;99m"
    BLUE = "\033[38;5;33m"
    CYAN = "\033[38;5;45m"
    GREEN = "\033[38;5;82m"
    YELLOW = "\033[38;5;221m"
    RED = "\033[38;5;196m"
    GRAY = "\033[38;5;240m"
    WHITE = "\033[38;5;255m"
    
    BG_DARK = "\033[48;5;235m"
    BG_PURPLE = "\033[48;5;55m"


def c(text, color=Colors.WHITE):
    return f"{color}{text}{Colors.RESET}"


# ============================================================
# CLI APP
# ============================================================

def print_header(engine):
    """Print the app header."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    model_info = engine.backend.get_info()
    model_status = "DEMO" if model_info.get("demo") else (
        "MODEL" if model_info.get("loaded") else "N/A"
    )
    
    print()
    print(c("╔══════════════════════════════════════════════╗", Colors.PURPLE))
    print(c("║", Colors.PURPLE) + c("     NEURA Forge Chat v1", Colors.BOLD) + 
          c("              ║", Colors.PURPLE))
    print(c("╚══════════════════════════════════════════════╝", Colors.PURPLE))
    print()
    print(f"  {c('✦', Colors.CYAN)} Modell: {c(model_status, Colors.GREEN if model_status == 'MODEL' else Colors.YELLOW)}")
    print(f"  {c('✦', Colors.CYAN)} Személyiség: {c(engine.current_personality, Colors.BLUE)}")
    print(f"  {c('✦', Colors.CYAN)} Beszélgetések: {c(str(len(engine.list_conversations())), Colors.BLUE)}")
    print()
    print(c("  Írj /help a parancsokért, /exit a kilépéshez.", Colors.GRAY))
    print()


def print_message(role, content):
    """Print a formatted message."""
    timestamp = time.strftime("%H:%M")
    
    if role == "user":
        print(f"  {c('┌─', Colors.BLUE)} {c('Te', Colors.BOLD)} {c(f'({timestamp})', Colors.GRAY)}")
        for line in content.split("\n"):
            print(f"  {c('│', Colors.BLUE)} {line}")
        print(f"  {c('└─', Colors.BLUE)}")
    else:
        print(f"  {c('┌─', Colors.PURPLE)} {c('NEURA', Colors.BOLD, Colors.PURPLE)} {c(f'({timestamp})', Colors.GRAY)}")
        for line in content.split("\n"):
            print(f"  {c('│', Colors.PURPLE)} {line}")
        print(f"  {c('└─', Colors.PURPLE)}")
    
    print()


def main():
    """Main CLI loop."""
    from models.forge_model import create_backend
    
    print()
    print(c("NEURA modell betöltése...", Colors.GRAY))
    
    backend = create_backend()
    engine = ChatEngine(backend=backend)
    
    print_header(engine)
    
    while True:
        try:
            # Input
            print(c("  ── ", Colors.GRAY) + c("Te", Colors.BOLD, Colors.BLUE) + 
                  c(" ── ", Colors.GRAY), end="")
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print(f"\n  {c('Viszlát!', Colors.PURPLE)} {c('✦', Colors.CYAN)}\n")
                break
            
            if user_input == "/refresh":
                print_header(engine)
                continue
            
            # Regular message
            print()
            response = ""
            
            # Show typing indicator
            stop_typing = threading.Event()
            
            def show_typing():
                while not stop_typing.is_set():
                    print(f"  {c('┌─', Colors.PURPLE)} {c('NEURA', Colors.BOLD, Colors.PURPLE)} {c('gépel...', Colors.GRAY)}", 
                          end="\r", flush=True)
                    stop_typing.wait(0.5)
                    if stop_typing.is_set():
                        break
                    print(f"  {c('┌─', Colors.PURPLE)} {c('NEURA', Colors.BOLD, Colors.PURPLE)} {c('gépel... ', Colors.GRAY)}", 
                          end="\r", flush=True)
                    stop_typing.wait(0.5)
            
            typing_thread = threading.Thread(target=show_typing, daemon=True)
            typing_thread.start()
            
            # Generate
            for chunk in engine.send_message(user_input, stream=True):
                response += chunk
            
            stop_typing.set()
            
            # Print response
            print_message("assistant", response)
            
        except KeyboardInterrupt:
            print(f"\n  {c('Viszlát!', Colors.PURPLE)} {c('✦', Colors.CYAN)}\n")
            break
        except EOFError:
            break
        except Exception as e:
            print(f"\n  {c(f'Hiba: {e}', Colors.RED)}\n")


if __name__ == "__main__":
    main()
