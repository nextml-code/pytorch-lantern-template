#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/env.sh'

cd "$PROJECT_PATH"

echo $PWD
poetry run autopep8 . --diff --exit-code --recursive
