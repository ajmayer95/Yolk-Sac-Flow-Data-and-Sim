#!/usr/bin/env python
"""Generate post-run segment_stitch metrics and visual QC report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from segment_stitch.generate_qc_report import main


if __name__ == "__main__":
    main()
