#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/env.sh'

cd "$PROJECT_PATH"

poetry run pytest
