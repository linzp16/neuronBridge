# Contributing

## Branches

- `main` contains reviewed and validated changes.
- `feature/<name>` contains new functionality.
- `fix/<name>` contains defect fixes.
- `release/<version>` contains release stabilization work.
- `hotfix/<name>` contains urgent fixes for a published version.

Do not push large feature changes directly to `main`.

## Commits

Keep commits focused and use one of these prefixes:

- `feat:` user-visible functionality
- `fix:` defect correction
- `refactor:` behavior-preserving restructuring
- `test:` test-only changes
- `docs:` documentation-only changes
- `build:` build or dependency changes
- `chore:` repository maintenance

Review staged changes before every commit:

```powershell
git diff --cached --name-status
git diff --cached --check
```

## Generated code

Model specifications, generated C++/CUDA files, registry changes, and their
baseline tests belong in the same pull request. Do not hand-edit generated
files unless the generator contract explicitly permits it.

## Release safety

Release artifacts must be built from a clean, tagged commit. Build trees,
virtual environments, dependency extraction directories, and release outputs
must not be committed.

Before opening a release pull request, run the Release CTest matrix and the
installed-wheel validator. `scripts/check_release_metadata.py` must pass;
public releases additionally require `--require-license`.
