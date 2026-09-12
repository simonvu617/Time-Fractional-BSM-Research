# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Run the collector with python -m tfbsm_collector."""

import sys

from tfbsm_collector import cli

if __name__ == "__main__":
    sys.exit(cli.main())
