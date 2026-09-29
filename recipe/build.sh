#!/usr/bin/env bash
set -e

shellcheck bin/check-glibc

# The source tree has no .git (rattler-build copies a plain file tree), so setuptools_scm
# needs the version handed to it explicitly rather than deriving it from `git describe`.
export SETUPTOOLS_SCM_PRETEND_VERSION="${PKG_VERSION}"
${PYTHON} -m pip install --no-deps --no-build-isolation -vv .
