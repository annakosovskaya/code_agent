from __future__ import annotations

import sys

from examples.demo_minimal import main


if __name__ == "__main__":
    # Thin wrapper preserved for backward compatibility.
    # Prefer running: python examples/demo_minimal.py
    sys.exit(main())


