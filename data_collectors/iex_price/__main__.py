"""Command line entry point for the IEX price collector.

Run it as a module so the package-relative imports resolve:

    python -m data_collectors.iex_price --start 2016-12-12 --end 2016-12-14
    python -m data_collectors.iex_price --start 2016-12-12 --end 2016-12-12 \
        --out bars.parquet
"""

import argparse
from pathlib import Path

from .collector import IEXPriceCollector


def main() -> None:
    parser = argparse.ArgumentParser(
        prog='python -m data_collectors.iex_price',
        description='Download IEX TOPS captures and write daily OHLCV bars to parquet.',
    )
    parser.add_argument(
        '--start', required=True, help='earliest session, YYYY-MM-DD')
    parser.add_argument(
        '--end', required=True, help='latest session, YYYY-MM-DD')
    parser.add_argument(
        '--out', type=Path,
        help='parquet file to write '
             '(default: ./iex_price_<start>_<end>.parquet)')
    args = parser.parse_args()

    out = args.out or Path.cwd() / f'iex_price_{args.start}_{args.end}.parquet'
    # Make the directory before the download, not after: a missing parent should
    # not surface only once there are hours of work to lose.
    out.parent.mkdir(parents=True, exist_ok=True)

    bars = IEXPriceCollector().collect(args.start, args.end)
    bars.to_parquet(out, index=False)

    print(f'wrote {len(bars):,} rows to {out}')


if __name__ == '__main__':
    main()
