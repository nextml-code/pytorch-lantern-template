#!/usr/bin/env sh

set -o errexit
set -o nounset

# Creating a test directory:
readonly TEST_DIR='.test-template'

# Scaffold the project:
readonly REPOSITORY_NAME='mnist-template'

readonly PROJECT_PATH="$TEST_DIR/$REPOSITORY_NAME"

# Exporting variables:
export REPOSITORY_NAME
