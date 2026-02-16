#!/bin/sh
set -eu

# Debug: Show LD_LIBRARY_PATH and check for SQLite
echo "=== Runtime Environment ==="
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-not set}"
echo "Checking for SQLite libraries:"
ls -lh /opt/lib/libsqlite3* 2>/dev/null || echo "No SQLite in /opt/lib"
echo "System SQLite:"
ls -lh /usr/lib64/libsqlite3* 2>/dev/null || echo "No SQLite in /usr/lib64"
echo "========================="

DEFAULT_ARGS="--cov-report term --cov-report xml:/var/task/tests/cobertura.xml --cov=/var/task /var/task/tests -vv"
exec python -m pytest $DEFAULT_ARGS "$@"
