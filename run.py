"""
NEURA Forge Chat v1 — Launch
"""
import os, sys, argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="NEURA Forge Chat v1 — Hungarian AI Chat"
    )
    parser.add_argument("mode", nargs="?", default="web",
                       choices=["web", "cli", "both"],
                       help="Interface mode (default: web)")
    parser.add_argument("--port", type=int, default=5000,
                       help="Web server port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1",
                       help="Web server host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true",
                       help="Flask debug mode")
    parser.add_argument("--demo", action="store_true",
                       help="Force demo mode (no model)")
    parser.add_argument("--model", type=str,
                       help="Path to custom model checkpoint")
    parser.add_argument("--assistant", action="store_true",
                       help="Use assistant checkpoint")
    
    args = parser.parse_args()
    
    # Config
    config = {}
    if args.demo:
        config["demo_mode"] = True
    if args.model:
        config["model_path"] = args.model
    if args.assistant:
        config["use_assistant"] = True
    
    # Run
    if args.mode == "cli":
        from cli import main as cli_main
        cli_main()
    elif args.mode == "web":
        from web.app import run as web_run
        web_run(host=args.host, port=args.port, debug=args.debug, config=config)
    elif args.mode == "both":
        import threading
        from web.app import run as web_run
        
        web_thread = threading.Thread(
            target=web_run,
            args=(args.host, args.port, args.debug, config),
            daemon=True
        )
        web_thread.start()
        
        from cli import main as cli_main
        cli_main()


if __name__ == "__main__":
    main()
