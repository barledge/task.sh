# Publishing

## crates.io Checklist

1. Ensure `Cargo.toml` metadata is complete (authors, description, repository, license).
2. Run `cargo login` with your crates.io API token.
3. Run tests in release mode:
   ```bash
   cargo test --release
   ```
4. Package and verify:
   ```bash
   cargo package --allow-dirty
   cargo publish --dry-run
   ```
5. Publish when ready:
   ```bash
   cargo publish
   ```

## GitHub Releases

Use the provided CI workflow (`.github/workflows/release.yml`) to build artifacts. Trigger by creating a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The workflow uploads binaries for macOS, Linux, and Windows.
