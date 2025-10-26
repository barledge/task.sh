#!/usr/bin/env bash
set -euo pipefail

# Packaging helper for crates.io
cargo package --allow-dirty

echo "Inspect target/package/ for the .crate file, then run:\n  cargo publish" 
