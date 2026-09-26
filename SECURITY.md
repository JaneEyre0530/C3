# Security and data boundary

The public C3 repository must remain free of credentials and private
infrastructure details. Report a suspected leak privately to the repository
maintainers rather than opening an issue with the secret in the body.

Before every release:

1. Run `python scripts/check_public_snapshot.py`.
2. Inspect new files for credentials, internal hostnames, absolute mounted
   paths, checkpoints, datasets, and rollout/evaluation dumps.
3. Keep service clients, cluster launchers, and proprietary environments in a
   private integration repository. Pass only serialized event spans into this
   package.

The scanner is a guardrail, not a substitute for human review. Rotate any
credential that was ever committed, even if it is later deleted.
