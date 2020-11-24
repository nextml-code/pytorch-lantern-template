#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/env.sh'

cd "$PROJECT_PATH"

echo $PWD
poetry run black \{\{cookiecutter.repository_name\}\} --check
