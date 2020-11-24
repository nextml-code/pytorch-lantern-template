#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/env.sh'

mkdir -p "$TEST_DIR" && cd "$TEST_DIR"

cookiecutter ../. \
    --no-input --overwrite-if-exists \
    repository_name="$REPOSITORY_NAME" \
    package_name="$PACKAGE_NAME" \
    package_version="$PACKAGE_VERSION" \
    package_description="$PACKAGE_DESCRIPTION"

cd "$REPOSITORY_NAME"
poetry install
