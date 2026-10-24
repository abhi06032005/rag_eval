import argparse
import os
import sys

from ragbench.data.un import UN_PERMISSION_ERROR_MSG, download_un_records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire UN documents (requires explicit permission reference)."
    )
    parser.add_argument("--limit", type=int, default=5000, help="Target document count")
    parser.add_argument("--output", type=str, default="data/raw/un", help="Output directory")
    parser.add_argument(
        "--acknowledge-permission", action="store_true", help="Acknowledge valid reuse permission"
    )
    args = parser.parse_args()

    permission_ref = os.environ.get("UN_PERMISSION_REFERENCE", "").strip()

    if not permission_ref or not args.acknowledge_permission:
        print(UN_PERMISSION_ERROR_MSG, file=sys.stderr)
        sys.exit(1)

    try:
        count = download_un_records(
            limit=args.limit,
            output_dir=args.output,
            permission_reference=permission_ref,
            acknowledge_permission=args.acknowledge_permission,
        )
        print(f"Successfully acquired {count} UN records under permission {permission_ref}")
    except PermissionError:
        sys.exit(1)


if __name__ == "__main__":
    main()
