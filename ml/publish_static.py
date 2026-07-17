"""Publish or verify the generated GitHub Pages directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from static_site import check_static_dashboard, publish_static_dashboard


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.check:
        check_static_dashboard(root)
        print("Static source and generated docs are in parity.")
        return
    publish_static_dashboard(root, root / "outputs" / "results.json")
    print("Generated docs/ from webapp/static/ and outputs/results.json.")


if __name__ == "__main__":
    main()
