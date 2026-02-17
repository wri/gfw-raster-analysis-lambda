#!/bin/sh
set -eu

DEFAULT_ARGS="--cov-report term --cov-report xml:/var/task/tests/cobertura.xml --cov=/var/task"
exec python -m pytest $DEFAULT_ARGS "$@"
