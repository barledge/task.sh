# Cross-compilation

The project supports producing binaries for macOS, Linux, and Windows using [cargo-zigbuild](https://github.com/rust-cross/cargo-zigbuild).

## Setup

```bash
cargo install cargo-zigbuild
```

## Build targets

```bash
scripts/build-cross.sh x86_64-unknown-linux-gnu
scripts/build-cross.sh x86_64-pc-windows-gnu
scripts/build-cross.sh aarch64-apple-darwin
```

Generated artifacts live under `target/<target>/release/task` (or `task.exe`).

Add Zig via your package manager (e.g., `brew install zig`).
