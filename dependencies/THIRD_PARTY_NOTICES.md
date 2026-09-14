# Third-party dependency notice

The Windows compatibility bundle is derived from validated Pinocchio and
ZeroMQ prefixes used by the historical `network_release` project.

Primary components currently detected are:

| Component | Version | Declared license |
|---|---:|---|
| Pinocchio | 4.0.0 | BSD-2-Clause |
| Boost | 1.88.0 | BSL-1.0 |
| Eigen | 3.4.0 | MPL-2.0 |
| ZeroMQ | 4.3.5 | MPL-2.0 |

Pinocchio has a substantial transitive dependency graph. The generated Conda
package inventory records package names, versions, builds, channels, and
declared license identifiers for audit purposes.

This inventory is not a substitute for the complete license texts. Public
redistribution of the compatibility bundle remains blocked until all included
components have complete license notices and redistribution terms have been
reviewed.
