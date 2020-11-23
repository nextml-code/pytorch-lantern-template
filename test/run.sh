#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/create.sh'

cd "$PROJECT_PATH"

poetry install
poetry add ../../test/wildfire-0.0.0-py3-none-any.whl
poetry run guild run prepare -y
poetry run guild run train max_epochs=2 n_batches_per_epoch=5 -y
poetry run guild run retrain max_epochs=2 n_batches_per_epoch=5 -y
