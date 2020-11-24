#!/usr/bin/env sh

set -o errexit
set -o nounset

. './test/env.sh'

cd "$PROJECT_PATH"

poetry run guild run prepare -y
poetry run guild run train max_epochs=2 n_batches_per_epoch=5 -y
poetry run guild run retrain max_epochs=2 n_batches_per_epoch=5 -y
