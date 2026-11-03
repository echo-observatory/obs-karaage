"""Command line: `karaage <command>`."""

import argparse
from pathlib import Path

import karaage
from karaage import __version__
from karaage.config import profile_from_env
from karaage.profile import load_profile


def download_models(embedding_model: str) -> None:
    """Fetch the models that are not installed as Python packages.

    The spaCy model `en_core_web_md` is a dependency of the package and is
    already installed. The benepar parser and the embedding model are
    downloaded to their libraries' usual caches (nltk_data and Hugging Face).
    """
    import benepar
    from sentence_transformers import SentenceTransformer
    from karaage.segmentation import load_nlp_pipeline

    print("Downloading the benepar model benepar_en3...")
    benepar.download("benepar_en3")
    print("Loading the NLP pipeline, which also fetches the tokenizer benepar uses...")
    load_nlp_pipeline()
    print(f"Downloading the embedding model {embedding_model}...")
    SentenceTransformer(embedding_model)
    print("All models are ready.")


def run(args, parser) -> int:
    """`karaage run`: one output per document of a JSONL file. Exit code 1 if a document failed."""
    from karaage.batch import InputError, parse_fields, read_documents, run_documents
    try:
        documents = read_documents(args.input, parse_fields(args.field))
    except (InputError, OSError) as e:
        parser.error(str(e))
    resources = karaage.setup(profile=args.profile)
    statuses = run_documents(documents, args.output, resources, report=args.report,
                             paraphrase=not args.no_paraphrase)
    return 1 if statuses["failed"] else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="karaage",
        description="Extract attributable statements from English texts.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser(
        "download-models",
        help="download the NLP and embedding models Karaage needs",
    )
    download.add_argument(
        "--profile",
        help="profile whose embedding model to download: a shipped profile name or a file "
             "(default: KARAAGE_PROFILE, or the default profile)",
    )

    run_parser = commands.add_parser(
        "run",
        help="run Karaage on a JSONL file of documents",
        description="Run Karaage on a JSONL file, one document per line with an id, a text, and optionally "
                    "a title and a published_at date. Writes <id>.json per document, and <id>.html with --report. "
                    "Documents whose output is already ok are skipped, so running the same command again "
                    "resumes an interrupted run and retries the documents that failed. The backend comes "
                    "from KARAAGE_* environment variables or a .env file.",
    )
    run_parser.add_argument("--input", required=True, type=Path, help="the JSONL file")
    run_parser.add_argument("--output", required=True, type=Path, help="directory for the output files")
    run_parser.add_argument(
        "--profile",
        help="a shipped profile name or a file (default: KARAAGE_PROFILE, or the default profile)",
    )
    run_parser.add_argument("--report", action="store_true", help="also write the HTML report of each document")
    run_parser.add_argument(
        "--no-paraphrase", action="store_true",
        help="don't paraphrase statements too close to the source's wording, nor reject them for it; "
             "only for sources whose wording may be reproduced, e.g. in the public domain, never for Retriever data",
    )
    run_parser.add_argument(
        "--field", action="append", default=[], metavar="FIELD=NAME",
        help="read FIELD (id, text, title or published_at) from the input field NAME, "
             "e.g. --field published_at=date; can be repeated",
    )

    args = parser.parse_args(argv)
    if args.command == "download-models":
        profile = load_profile(args.profile or profile_from_env())
        download_models(profile.embedding_model)
    elif args.command == "run":
        return run(args, run_parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
