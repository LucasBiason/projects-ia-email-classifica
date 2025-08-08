#!/usr/bin/env bash

set -ef

cli_help() {
  cli_name=${0##*/}
  echo "
$cli_name
System entrypoint cli
Usage: $cli_name [command]
Commands:
  runserver     deploy runserver
  *             Help
"
  exit 1
}

case "$1" in
  test)
    poetry run python -m pytest tests/ -v
    ;;
  test-with-coverage)
    poetry run coverage run -m pytest --cov=app --cov-report term-missing --cov-config=.coveragerc tests/ -s -vv
    ;;
  runserver)
    poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  *)
    cli_help
    ;;
esac
