#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/create.sh'

cd "$PROJECT_PATH"

poetry install
poetry run guild run train max_epochs=2
