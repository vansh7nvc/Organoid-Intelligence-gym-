"""
CLI Launcher for the OrganoidEnv Interactive Platform.

Usage:
    py -3.11 scripts/launch_platform.py --port 8000
"""

import argparse
import os
import sys
import webbrowser

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    parser = argparse.ArgumentParser(description="Launch OrganoidEnv Interactive Web Platform")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Starting OrganoidEnv Interactive Platform at {url}")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run("organoid_rl.platform.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
