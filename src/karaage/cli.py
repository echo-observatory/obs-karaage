"""Command line: `karaage <command>`."""

import argparse

from karaage import __version__

# The embedding model of the pre-1.1.0 setup. Model profiles will replace this default.
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"


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


def main(argv=None) -> None:
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
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"embedding model to download (default: {DEFAULT_EMBEDDING_MODEL})",
    )

    args = parser.parse_args(argv)
    if args.command == "download-models":
        download_models(args.embedding_model)


if __name__ == "__main__":
    main()
