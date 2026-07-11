"""
NEURA Forge Chat v1 — Web Interface
FastAPI/Flask web app with SSE streaming
"""
import os, sys, json, time, uuid
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from engine.chat_engine import ChatEngine, PERSONALITIES

app = Flask(__name__)
engine = None


def init_engine(config: dict = None):
    """Initialize the chat engine (call once at startup)."""
    global engine
    if engine is None:
        from models.forge_model import create_backend
        backend = create_backend(config)
        engine = ChatEngine(backend=backend)
    return engine


# ============================================================
# API Routes
# ============================================================

@app.route("/")
def index():
    """Main chat page."""
    return render_template("chat.html", 
                         personalities=engine.get_personalities() if engine else [])


@app.route("/api/info")
def api_info():
    """Get system info."""
    return jsonify({
        "model": engine.backend.get_info() if engine else {"loaded": False},
        "personalities": engine.get_personalities() if engine else [],
        "current_personality": engine.current_personality if engine else "default",
        "current_conv": engine.current_conv_id if engine else None,
    })


@app.route("/api/conversations")
def api_conversations():
    """List conversations."""
    if not engine:
        return jsonify([])
    return jsonify(engine.list_conversations())


@app.route("/api/conversations/<conv_id>")
def api_conversation(conv_id):
    """Get conversation messages."""
    if not engine:
        return jsonify({"messages": []})
    conv = engine.switch_conversation(conv_id)
    messages = engine.db.get_messages(conv_id)
    return jsonify({
        "conversation": conv,
        "messages": messages,
    })


@app.route("/api/conversations", methods=["POST"])
def api_new_conversation():
    """Create new conversation."""
    if not engine:
        return jsonify({"error": "Engine not initialized"}), 500
    
    data = request.get_json(silent=True) or {}
    personality = data.get("personality", engine.current_personality)
    conv = engine.new_conversation(personality=personality)
    return jsonify(conv)


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
def api_delete_conversation(conv_id):
    """Delete conversation."""
    if not engine:
        return jsonify({"error": "Engine not initialized"}), 500
    engine.delete_conversation(conv_id)
    return jsonify({"status": "deleted"})


@app.route("/api/personality", methods=["POST"])
def api_set_personality():
    """Set personality."""
    if not engine:
        return jsonify({"error": "Engine not initialized"}), 500
    
    data = request.get_json(silent=True) or {}
    name = data.get("personality", "default")
    if engine.set_personality(name):
        return jsonify({"status": "ok", "personality": name})
    return jsonify({"error": "Unknown personality"}), 400


# ============================================================
# Streaming Chat
# ============================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Send message and stream response via SSE.
    
    Request: {"message": "...", "conv_id": "...", "personality": "..."}
    Response: SSE stream of text chunks
    """
    if not engine:
        return jsonify({"error": "Engine not initialized"}), 500
    
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    conv_id = data.get("conv_id")
    personality = data.get("personality")
    
    if not message:
        return jsonify({"error": "Empty message"}), 400
    
    # Switch conversation if specified
    if conv_id and conv_id != engine.current_conv_id:
        engine.switch_conversation(conv_id)
    
    # Set personality if specified
    if personality:
        engine.set_personality(personality)
    
    def generate():
        """SSE stream generator."""
        response = ""
        for chunk in engine.send_message(message, stream=True):
            response += chunk
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        
        yield f"data: {json.dumps({'done': True, 'full': response})}\n\n"
        yield f"data: {json.dumps({'conv_id': engine.current_conv_id})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================================
# Main
# ============================================================

def run(host="127.0.0.1", port=5000, debug=False, config: dict = None):
    """Run the web server."""
    init_engine(config)
    
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║     NEURA Forge Chat v1 — Web Interface     ║")
    print("╚══════════════════════════════════════════════╝")
    print()
    print(f"  🌐 Web:     http://{host}:{port}")
    print(f"  📡 API:     http://{host}:{port}/api/info")
    print(f"  💬 Chat:    http://{host}:{port}/")
    print()
    print("  Ctrl+C to stop")
    print()
    
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run()
