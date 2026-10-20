#!/usr/bin/env python3
"""
Launcher script for SafeGuard AI website.
Starts the server and automatically opens the user's web browser.
"""

import sys
import webbrowser
import threading
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from website.server import start_server

def open_browser(port):
    time.sleep(1.2)
    url = f"http://localhost:{port}"
    print(f"🚀 Opening browser at {url}...")
    webbrowser.open(url)

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    browser_thread = threading.Thread(target=open_browser, args=(port,), daemon=True)
    browser_thread.start()

    start_server(port)
