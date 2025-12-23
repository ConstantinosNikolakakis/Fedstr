#!/usr/bin/env python3
"""
Simple HTTP File Server for FEDSTR Model Storage
=================================================

Provides PUT and GET endpoints for storing/retrieving model files.

Usage:
    python3 file_server.py [--port 8000] [--dir /tmp/fedstr_models]

Endpoints:
    PUT  /model_abc123.bin  - Upload a model
    GET  /model_abc123.bin  - Download a model
    GET  /                  - List all models
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import argparse
import json
from datetime import datetime
from pathlib import Path


class ModelStorageHandler(BaseHTTPRequestHandler):
    """HTTP handler for model file storage"""
    
    def do_GET(self):
        """Serve files or list directory"""
        if self.path == '/':
            # List all models
            self.list_models()
        else:
            # Serve specific file
            self.serve_file()
    
    def do_PUT(self):
        """Accept file uploads"""
        try:
            # Get filename from path
            filename = self.path.lstrip('/')
            if not filename:
                self.send_error(400, "Filename required")
                return
            
            # Read uploaded data
            content_length = int(self.headers.get('Content-Length', 0))
            file_data = self.rfile.read(content_length)
            
            # Save to storage directory
            filepath = os.path.join(self.server.storage_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(file_data)
            
            # Build response URL
            url = f"http://{self.headers.get('Host', 'localhost')}{self.path}"
            
            # Send success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'status': 'success',
                'url': url,
                'filename': filename,
                'size': len(file_data),
                'timestamp': datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())
            
            print(f"✓ Uploaded: {filename} ({len(file_data)} bytes)")
            
        except Exception as e:
            print(f"✗ Upload error: {e}")
            self.send_error(500, str(e))
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, PUT, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def serve_file(self):
        """Serve a specific file"""
        try:
            filename = self.path.lstrip('/')
            filepath = os.path.join(self.server.storage_dir, filename)
            
            if not os.path.exists(filepath):
                self.send_error(404, f"File not found: {filename}")
                return
            
            # Read and serve file
            with open(filepath, 'rb') as f:
                data = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/octet-stream')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(data)
            
            print(f"→ Served: {filename} ({len(data)} bytes)")
            
        except Exception as e:
            print(f"✗ Serve error: {e}")
            self.send_error(500, str(e))
    
    def list_models(self):
        """List all stored models"""
        try:
            models = []
            storage_path = Path(self.server.storage_dir)
            
            for filepath in storage_path.glob('*'):
                if filepath.is_file():
                    stat = filepath.stat()
                    models.append({
                        'name': filepath.name,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'url': f"http://{self.headers.get('Host', 'localhost')}/{filepath.name}"
                    })
            
            # Sort by modification time (newest first)
            models.sort(key=lambda x: x['modified'], reverse=True)
            
            # Send JSON response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                'status': 'success',
                'count': len(models),
                'models': models,
                'storage_dir': str(storage_path.absolute())
            }
            self.wfile.write(json.dumps(response, indent=2).encode())
            
        except Exception as e:
            print(f"✗ List error: {e}")
            self.send_error(500, str(e))
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        # Don't log every request (too noisy)
        pass


class ModelStorageServer(HTTPServer):
    """HTTP server with storage directory attribute"""
    def __init__(self, server_address, RequestHandlerClass, storage_dir):
        self.storage_dir = storage_dir
        super().__init__(server_address, RequestHandlerClass)


def main():
    parser = argparse.ArgumentParser(description='FEDSTR Model Storage Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    parser.add_argument('--dir', default='/tmp/fedstr_models', help='Storage directory')
    args = parser.parse_args()
    
    # Create storage directory
    os.makedirs(args.dir, exist_ok=True)
    
    # Start server
    server_address = ('', args.port)
    httpd = ModelStorageServer(server_address, ModelStorageHandler, args.dir)
    
    print("╔═══════════════════════════════════════╗")
    print("║  FEDSTR Model Storage Server          ║")
    print("╚═══════════════════════════════════════╝")
    print(f"\n📁 Storage directory: {args.dir}")
    print(f"🌐 Server address:    http://localhost:{args.port}")
    print(f"\nEndpoints:")
    print(f"  GET  /               - List all models")
    print(f"  GET  /model_xyz.bin  - Download model")
    print(f"  PUT  /model_xyz.bin  - Upload model")
    print(f"\nExample usage:")
    print(f"  # Upload")
    print(f"  curl -X PUT --data-binary @model.bin http://localhost:{args.port}/model_abc.bin")
    print(f"  # Download")
    print(f"  curl http://localhost:{args.port}/model_abc.bin -o model.bin")
    print(f"  # List")
    print(f"  curl http://localhost:{args.port}/")
    print(f"\nPress Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped")


if __name__ == '__main__':
    main()
