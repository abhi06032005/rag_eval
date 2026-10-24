import argparse

from ragbench.data.govinfo import download_govinfo_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire GovInfo public domain documents.")
    parser.add_argument("--limit", type=int, default=15000, help="Target document count")
    parser.add_argument("--output", type=str, default="data/raw/govinfo", help="Output directory")
    parser.add_argument(
        "--snapshot-id", type=str, default="govinfo-bulk-2026", help="Snapshot identifier"
    )
    args = parser.parse_args()

    count = download_govinfo_records(
        limit=args.limit, output_dir=args.output, snapshot_id=args.snapshot_id
    )
    print(f"Successfully acquired {count} GovInfo records to {args.output}")


if __name__ == "__main__":
    main()
