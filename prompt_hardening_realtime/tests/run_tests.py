"""Run the standalone test suites shipped with this exported project."""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUITES = (
    "test_security_hardening_analyzer",
    "test_security_hardening_contracts",
    "test_security_hardening_workflow",
    "test_security_hardening_integrity",
    "test_source_sync",
    "test_directory_worker",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    loader = unittest.TestLoader()
    all_tests = unittest.TestSuite()
    for name in SUITES:
        all_tests.addTests(loader.discover(str(TEST_DIR), pattern=f"{name}.py", top_level_dir=str(ROOT)))
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(all_tests)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
