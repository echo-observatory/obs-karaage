# Commit checklist

## Before every commit

Check each item. If one fails, fix it or say so before committing.

- [ ] The branch name is `<issue-number>-short-description` and the change belongs to that one issue.
- [ ] The commit is atomic: one logical change, no unrelated edits or formatting churn.
- [ ] The message is short and imperative, with no agent attribution.
- [ ] Nothing staged looks like a secret, a `.env` file or raw data.
- [ ] `README.md` is still accurate (purpose, inputs, outputs, setup and commands) and updated in the same commit if the change affects it.
- [ ] If a schema or API changed, its spec or schema file is updated in the same commit.
- [ ] If the repo's role, inputs or outputs changed, its line in the `pipeline` README is updated (a separate PR in that repo).
- [ ] The code runs, or the existing tests pass.

## Before tagging a release

- [ ] The version bump follows the rules in `AGENTS.md` (MAJOR, MINOR or PATCH).
- [ ] The tag `vX.Y.Z` does not exist yet. Existing tags are never moved or deleted.
- [ ] The tag goes on `main`, on a commit whose CI passed.

## Before publishing a release

- [ ] The release workflow passed and created the draft release, with the wheel, the sdist and the requirements file.
- [ ] The reference run is attached: the release's version, with the default profile, run live on the test dataset (`karaage run --report`), outputs and reports zipped together.
- [ ] The release notes list the shipped profiles.

## Before opening a PR

- [ ] The PR description references the issue (`Closes #<number>`).
- [ ] The diff has been skimmed by the human who is committing. They are responsible for the code.
