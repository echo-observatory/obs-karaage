# AGENTS.md

## About this repo

Karaage extracts attributable statements from English news articles and rephrases them, each with a reference to the passage it comes from.
It is the "Infer statements" step of the pipeline, and its output is meant to go into snapshot bundles (see `obs-corpus-snapshot`).

## Pipeline context

The `pipeline` repo's README is the single source for how the components fit together.
Read `../pipeline/README.md` before any work that touches another component or a shared interface.
If that file does not exist, tell the user that the `pipeline` repo needs to be cloned next to this one.
Do not copy its content into this repo.

## Conventions

- Branch from `main` as `<issue-number>-short-description`. Never commit directly to `main`.
- Commit messages are short and imperative. No agent attribution (no `Co-Authored-By`, no "Generated with ..." lines) in commits or PRs.
- Never commit secrets, `.env` files or raw corpus data (the test datasets below are the exception).
- Repo and file names are lowercase-kebab.
- Before every commit and before opening a PR, read and follow `COMMIT_CHECKLIST.md`.

## Data

This repo will be public, including its history.

- The only source data in this repo is `data/test-datasets/`: English Wikinews articles that may be redistributed (licences in each dataset's README). Outputs derived from them may be committed.
- Nothing derived from other corpora goes in: no logs, results or reports from runs on them, and no excerpts of their text in code, prompts, comments or tests. Use invented examples.
- Retriever data must never be sent to OpenRouter or any other external API. This is a team rule; the code does not enforce it.

## Versions and releases

- Versions are `MAJOR.MINOR.PATCH`. MAJOR: the output format breaks. MINOR: any change that can change the extracted statements (code, prompts, parameters, models, dependencies). PATCH: no change to the output.
- A released version is never changed. Never move or delete a `v*` tag; fix a mistake with a new version.
- Version history starts at 1.1.0. Earlier versions (up to 1.0.2) are not part of this repo.

## Documentation

`README.md` is the documentation for this repo: purpose, inputs, outputs, and how to set up, run and test.
Read it before working here, and keep it up to date as part of every change that affects it.
Do not repeat that content in this file.

Until 1.1.0 is released, also read `docs/design-1.1.0.md`: it holds the decisions every 1.1.0 issue relies on.
