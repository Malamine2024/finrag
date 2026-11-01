"""Command-line interface:  python -m finrag.ask "What was the net income in 2024?" """
import argparse

from .config import TOP_K
from .generate import answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the annual reports.")
    parser.add_argument("question")
    parser.add_argument("--company", help="filter on a company, e.g. banque-nationale")
    parser.add_argument("--year", type=int, help="filter on a fiscal year")
    parser.add_argument("-k", type=int, default=TOP_K, help="number of chunks to retrieve")
    args = parser.parse_args()

    result = answer(args.question, k=args.k, company=args.company, year=args.year)
    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for i, hit in enumerate(result["sources"], start=1):
        m = hit["metadata"]
        print(f"  [S{i}] {m['source']} p.{m['page']} ({m['kind']}, distance {hit['distance']:.3f})")


if __name__ == "__main__":
    main()
