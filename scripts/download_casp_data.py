#!/usr/bin/env python3
"""Download CASP datasets.

Usage:
    python scripts/download_casp_data.py --casp-version 14 15 16
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.casp_benchmark import CASPDataset


def main():
    parser = argparse.ArgumentParser(description='Download CASP datasets')
    parser.add_argument('--casp-version', type=int, nargs='+',
                       default=[14], choices=[14, 15, 16],
                       help='CASP versions to download')
    parser.add_argument('--data-dir', type=str, default='data/casp',
                       help='Directory to store data')
    
    args = parser.parse_args()
    
    print("CASP Dataset Downloader")
    print("="*80)
    
    for version in args.casp_version:
        print(f"\nDownloading CASP{version}...")
        dataset = CASPDataset(
            casp_version=version,
            data_dir=args.data_dir,
            download=True
        )
        print(f"CASP{version}: {len(dataset)} structures loaded")
    
    print("\nDownload complete!")


if __name__ == '__main__':
    main()
