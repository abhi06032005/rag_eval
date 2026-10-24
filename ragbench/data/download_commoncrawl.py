import argparse

from ragbench.data.commoncrawl import download_commoncrawl_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire allowlisted Common Crawl WET web passages."
    )
    parser.add_argument(
        "--crawl", type=str, default="CC-MAIN-2026-43", help="Common Crawl snapshot ID"
    )
    parser.add_argument("--limit", type=int, default=25000, help="Target document count")
    parser.add_argument(
        "--allowlist",
        type=str,
        default="configs/commoncrawl_allowlist.txt",
        help="Domain allowlist",
    )
    parser.add_argument(
        "--output", type=str, default="data/raw/commoncrawl", help="Output directory"
    )
    args = parser.parse_args()

    count = download_commoncrawl_records(
        limit=args.limit,
        output_dir=args.output,
        crawl_id=args.crawl,
        allowlist_path=args.allowlist,
    )
    print(f"Successfully processed {count} allowlisted Common Crawl records to {args.output}")


if __name__ == "__main__":
    main()
