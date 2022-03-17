#!/usr/bin/env sh

set -o errexit
set -o nounset

# Creating a test directory:
readonly TEST_DIR='test'

# Scaffold the project:
readonly REPOSITORY_NAME='cifar10'

readonly PROJECT_PATH="$TEST_DIR/$REPOSITORY_NAME"

# Exporting variables:
export REPOSITORY_NAME
