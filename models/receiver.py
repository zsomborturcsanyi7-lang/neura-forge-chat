"""File receiver: receives files via HTTP PUT and saves them."""
import os, sys
from http.server import HTTPServer, BaseHTTPRequestHandler

SAVE_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

class Receiver(BaseHTTPRequestHandler):
    def do_PUT(self):
        path = self.path.strip("/")
        if not path:
            self.send_response(400)
            self.end_headers()
            return
        
        save_path = os.path.join(SAVE_DIR, os.path.basename(path))
        length = int(self.headers.get('Content-Length', 0))
        
        with open(save_path, 'wb') as f:
            f.write(self.rfile.read(length))
        
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"OK {os.path.getsize(save_path)}".encode())
    
    def log_message(self, format, *args):
        pass  # quiet

port = int(sys.argv[2]) if len(sys.argv) > 2 else 9997
server = HTTPServer(('0.0.0.0', port), Receiver)
print(f"Receiver on :{port}, saving to {SAVE_DIR}")
server.serve_forever()
