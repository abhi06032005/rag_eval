import argparse

from ragbench.data.wikipedia import download_wikipedia_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire English Wikipedia articles.")
    parser.add_argument("--limit", type=int, default=30000, help="Target document count")
    parser.add_argument("--output", type=str, default="data/raw/wikipedia", help="Output directory")
    parser.add_argument(
        "--snapshot-id", type=str, default="enwiki-20261001", help="Snapshot identifier"
    )
    args = parser.parse_args()

    count = download_wikipedia_records(
        limit=args.limit, output_dir=args.output, snapshot_id=args.snapshot_id
    )
    print(f"Successfully acquired {count} Wikipedia records to {args.output}")


if __name__ == "__main__":
    main()
