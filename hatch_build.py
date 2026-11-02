"""Build hook: record the git commit the package is built from.

A release version alone does not say which commit it was built from, and
outputs record that commit (docs/design-1.1.0.md, section 6.3).
"""

import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

BUILD_INFO = "src/karaage/_build_info.py"


class BuildInfoHook(BuildHookInterface):
    def initialize(self, version, build_data):
        path = Path(self.root, BUILD_INFO)
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.root, capture_output=True, text=True, check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        # Without git (e.g. building from an sdist), keep the file written when the sdist was made.
        if commit is not None or not path.exists():
            path.write_text(
                "# Written at build time by hatch_build.py; not tracked in git.\n"
                f"commit = {commit!r}\n",
                encoding="utf-8",
            )
        build_data["artifacts"].append(f"/{BUILD_INFO}")
