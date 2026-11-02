# obs-karaage

Karaage extracts attributable statements from English news articles and rephrases them, each with a reference to the passage it comes from.
It is the "Extract positions" step of the pipeline; see the `pipeline` repo for how the components fit together.

**Status:** being restructured into a versioned Python package. The first release will be 1.1.0.
Until then, the code is the pre-1.1.0 version and its interface will change.

## Inputs

One English text per document. Karaage does not translate: sources in other languages must be translated beforehand.

## Outputs

For each document, a JSON file with the main point of the text, the actors found in it, and the extracted statements, each with its actor, a rephrased statement, a one-sentence version and the range of segments it comes from.
Optionally, an HTML report for checking the results by hand.

For 1.1.0, the output will be aligned with the snapshot specification in `obs-corpus-snapshot`.

## Setup

`Pipeline usage.ipynb` shows how to install the dependencies, connect to an LLM served by LM Studio and run the pipeline on a batch of documents.
It predates the restructuring and will be rewritten.

## Test dataset

`data/test-datasets/wikinews-piracy-30/` holds 30 English Wikinews articles about piracy, file sharing and DRM, under CC BY 2.5 or in the public domain.
See its README for sources, licences and attribution.
