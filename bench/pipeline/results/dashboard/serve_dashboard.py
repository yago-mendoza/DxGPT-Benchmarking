#!/usr/bin/env python3
"""
Simple HTTP server for the experiment dashboard.
Serves files and enables CORS for local development.
"""

import http.server
import socketserver
import os
import json
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlparse

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        # Handle API endpoint for experiments
        if parsed_path.path == '/api/experiments':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            # Get the current directory (should be results/)
            current_dir = os.getcwd()
            
            # Find all experiment folders in current directory
            experiments = []
            if os.path.exists(current_dir):
                for folder in os.listdir(current_dir):
                    if folder.startswith('experiment_') and os.path.isdir(os.path.join(current_dir, folder)):
                        experiments.append(folder)
            
            # Sort experiments by name (newest first based on timestamp)
            experiments.sort(reverse=True)
            
            print(f"Found experiments: {experiments}")  # Debug info
            
            # Send JSON response
            self.wfile.write(json.dumps(experiments).encode())
        else:
            # Default file serving
            super().do_GET()

def main():
    PORT = 8000
    
    # Get the dashboard directory (where this script is located)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the results directory (parent of dashboard)
    results_dir = os.path.dirname(script_dir)
    os.chdir(results_dir)
    
    print(f"Starting server in: {os.getcwd()}")
    print(f"Dashboard script location: {script_dir}")
    
    # Verify dashboard file exists
    dashboard_file = os.path.join('scripts', 'index.html')
    if os.path.exists(dashboard_file):
        print(f"Dashboard file found: {dashboard_file}")
    else:
        print(f"WARNING: Dashboard file not found at {dashboard_file}")
    
    # List existing experiments for debugging
    experiments = [f for f in os.listdir('.') if f.startswith('experiment_') and os.path.isdir(f)]
    print(f"Available experiments: {experiments}")
    
    # Try to find an available port
    for port in range(PORT, PORT + 10):
        try:
            with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
                print(f"\n{'='*60}")
                print(f"🚀 Server running at http://localhost:{port}/")
                print(f"📊 Open http://localhost:{port}/dashboard/scripts/index.html")
                print(f"{'='*60}")
                print("\nPress Ctrl+C to stop the server")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\n\n👋 Server stopped.")
                break
        except OSError as e:
            if e.errno == 98:  # Address already in use
                print(f"Port {port} is already in use, trying next port...")
                continue
            else:
                raise

if __name__ == "__main__":
    main()