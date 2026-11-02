"""Command line: `karaage <command>`."""

import argparse

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
        "--profile",
        help="profile whose embedding model to download: a shipped profile name or a file "
             "(default: KARAAGE_PROFILE, or the default profile)",
    )

    args = parser.parse_args(argv)
    if args.command == "download-models":
        profile = load_profile(args.profile or profile_from_env())
        download_models(profile.embedding_model)


if __name__ == "__main__":
    main()
