"""Request a paired telemetry/video download from the local training FastAPI app."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import httpx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", help="Backend racing session ID (24 hexadecimal characters).")
    parser.add_argument("--api-url", default="http://127.0.0.1:8002", help="Local training API URL.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-fA-F]{24}", args.session_id):
        parser.error("session_id must contain 24 hexadecimal characters")
    token = os.environ.get("BACKEND_ACCESS_TOKEN", "").strip()
    if not token:
        parser.error("Set BACKEND_ACCESS_TOKEN to the session owner's backend JWT")

    try:
        # A session can contain several large recordings; the server applies
        # individual backend request timeouts while this request waits for the pair.
        with httpx.Client(timeout=httpx.Timeout(None, connect=10.0)) as client:
            response = client.post(
                f"{args.api_url.rstrip('/')}/racing-session/{args.session_id}/download",
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
        return 0
    except httpx.HTTPStatusError as exc:
        print(f"Download failed (HTTP {exc.response.status_code}): {exc.response.text}", file=sys.stderr)
    except httpx.RequestError:
        print(f"Cannot reach the training API at {args.api_url}. Start training.api.app first.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
