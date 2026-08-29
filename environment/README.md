# Clean-environment preflight

This preflight was performed on 2026-08-29 in a new repository-local `.venv`
using Python 3.12.0 on macOS arm64. It installed `requirements.txt` at the
existing pinned EMBER2024 revision, queried only lightweight upstream metadata,
calculated storage requirements, and ran the synthetic verification suite. It
did not download dataset archives or the benchmark model and did not run any
real project stage.

`requirements-lock.txt` is the exact `pip freeze --all` result after adding the
small compatibility pin `signify==0.7.1`. That pin is necessary because the
pinned `thrember` source imports `signify.authenticode.SignedPEFile`, which is
not exported by the initially resolved signify 0.9.2. The lock is a snapshot of
this macOS arm64 host environment, not a universal cross-platform lock file.

The Python dependency installation completed and `pip check` passed. The
initial preflight found that LightGBM could not locate `libomp.dylib`. When the
runtime-remediation check resumed, Homebrew 6.0.20 reported that `libomp 22.1.8`
was already installed at `/opt/homebrew/opt/libomp`; it was not reinstalled.
Fresh-process imports of LightGBM 4.7.0 and top-level `thrember 0.1.0` then
succeeded, completing runtime compatibility for this host.

The initial preflight loaded the installed pinned `features.py` directly,
without running vectorization, to inspect `PEFeatureExtractor().dim`. The
runtime-remediation check subsequently confirmed the same dimension, 2568,
through a normal top-level `thrember` import. No model or project data was
loaded.

See `preflight.json` for machine-readable versions, upstream revisions, file
sizes, disk calculations, checks, and unresolved items. It intentionally uses
only repository-relative paths and contains no user identity, credentials, or
private home-directory information.
