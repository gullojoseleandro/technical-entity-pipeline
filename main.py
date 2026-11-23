"""Run one asynchronous extraction example from the command line."""

import argparse
import asyncio
import logging

from chain import ExtractionError, process_text


EXAMPLE = (
    "A FastAPI service uses Redis for caching and PostgreSQL for storage. "
    "The production API is failing under concurrent database connections."
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract technical entities from text.")
    parser.add_argument("--text", default=EXAMPLE, help="Paragraph to analyze")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        result = asyncio.run(process_text(args.text))
    except (ExtractionError, ValueError) as error:
        logging.error("%s", error)
        return 1
    except Exception as error:
        logging.error("Provider call failed (%s)", type(error).__name__)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
