#!/bin/bash

set -e  # stop on first error

PACKAGE_NAME=$1

if [ -z "$PACKAGE_NAME" ]; then
  echo "Usage: $0 <package-name>"
  exit 1
fi

BASE_DIR=$(pwd)

echo "Upgrading $PACKAGE_NAME in root project..."
uv sync --upgrade-package "$PACKAGE_NAME"

echo "Upgrading $PACKAGE_NAME in mlops-kafka/app..."
cd "$BASE_DIR/mlops-kafka/app"
uv sync --upgrade-package "$PACKAGE_NAME"

echo "Upgrading $PACKAGE_NAME in mlops-cronjob..."
cd "$BASE_DIR/mlops-cronjob"
uv sync --upgrade-package "$PACKAGE_NAME"

echo "Done."
