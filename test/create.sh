#!/usr/bin/env sh

set -o errexit
set -o nounset

# Creating a test directory:
readonly TEST_DIR='.test-template'

# Scaffold the project:
readonly REPOSITORY_NAME='mnist-template'
readonly PACKAGE_NAME='mnist_template'
readonly PACKAGE_VERSION='0.0.1'
readonly PACKAGE_DESCRIPTION='Placeholder description'

readonly PROJECT_PATH="$TEST_DIR/$REPOSITORY_NAME"


mkdir -p "$TEST_DIR" && cd "$TEST_DIR"

cookiecutter ../. \
    --no-input --overwrite-if-exists \
    repository_name="$REPOSITORY_NAME" \
    package_name="$PACKAGE_NAME" \
    package_version="$PACKAGE_VERSION" \
    package_description="$PACKAGE_DESCRIPTION"

cd '..'


# Exporting variables:
export REPOSITORY_NAME
export PACKAGE_NAME
export PACKAGE_VERSION
export PACKAGE_DESCRIPTION
export PROJECT_PATH
