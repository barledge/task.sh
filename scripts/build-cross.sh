#!/usr/bin/env bash
set -euo pipefail

cargo zigbuild --release --target $1
