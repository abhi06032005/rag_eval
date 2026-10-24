import argparse

from ragbench.data.arxiv import download_arxiv_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire arXiv descriptive metadata and abstracts."
    )
    parser.add_argument("--limit", type=int, default=30000, help="Target document count")
    parser.add_argument("--output", type=str, default="data/raw/arxiv", help="Output directory")
    parser.add_argument(
        "--snapshot-id", type=str, default="arxiv-2026-10", help="Snapshot identifier"
    )
    args = parser.parse_args()

    count = download_arxiv_records(
        limit=args.limit, output_dir=args.output, snapshot_id=args.snapshot_id
    )
    print(f"Successfully acquired {count} arXiv records to {args.output}")


if __name__ == "__main__":
    main()
