#!/usr/bin/env bash
set -e

ls -lah
cp -r bin "$PREFIX/bin"
shellcheck "$PREFIX/bin/check-glibc"
