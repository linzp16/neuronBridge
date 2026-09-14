# Bundled dependencies

NeuronBridge will support `AUTO`, `BUNDLED`, and `SYSTEM` dependency modes.
The initial supported bundle target is `windows-x64-msvc`.

During the dependency migration stage, the validated Pinocchio and ZeroMQ
prefixes from the historical workspace will be audited, minimized, packaged,
hashed, and recorded in `manifest.json`.

The versioned archive belongs under:

```text
dependencies/bundles/windows-x64-msvc/
```

The archive is tracked through Git LFS. Its expanded contents belong under
`dependencies/vendor/` and are never committed as individual files.

The repository-bootstrap stage intentionally contains no dependency archive.
