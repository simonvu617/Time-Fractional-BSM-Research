# Copyright 2026 Simon Vu
# SPDX-License-Identifier: MIT

"""Run the ThetaData collector; use --help for collection options."""

import sys

from tfbsm_collector import cli

if __name__ == "__main__":
    sys.exit(cli.main())
