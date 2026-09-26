# C3

This repository contains the public, framework-independent implementation of
the C3 paper's actor-side verifier modulation for policy-gradient training.
It is intentionally an algorithm repository, not a copy of an internal
training workspace.

## What is included

- `c3/core.py`: token-span event handling, policy-confidence modes, and the C3
  actor-advantage update.
- `c3/events.py`: a model/parser-agnostic implementation of the three public
  event rules (invalid action, exact loop, near loop).
- `configs/c3_public.toml`: platform-neutral defaults.
- `tests/`: CPU tests for overlap handling, masking, confidence, clipping, and
  zero-advantage rescue.
- `examples/apply_c3.py`: a small trainer integration example.
- `scripts/check_public_snapshot.py`: a fail-closed pre-publication scan.
- `SOURCE_MAP.md`: the private-to-public responsibility boundary.

The package does not contain model weights, datasets, rollout traces,
checkpoints, internal URLs, cluster launchers, or credentials. A trainer should
provide verifier events in this shape:

```json
{
  "pattern": "invalid_action",
  "value": -1.0,
  "target": {"start": 120, "end": 180},
  "confidence_target": {"start": 120, "end": 160}
}
```

`target` and `confidence_target` use half-open token offsets. Event values are
negative severities in `[-1, 0]`; positive verifier events are ignored by C3.

## Algorithm contract

First compute the ordinary environment advantage `A_env` and keep the critic
return `R_env` unchanged. For each token covered by a bad event, let `q` be the
maximum event severity and `c` be the detached policy confidence. For positive
advantages:

```text
scale = q * (reversal_scale + confidence_reversal_scale * c)
A_actor = -(1 + scale) * A_env
R_critic = R_env
```

By default, non-positive advantages, clean tokens, and masked tokens are
unchanged. Overlapping events are order-independent. The reported C3
configuration uses `span_mean_prob`, the arithmetic mean of action-token
probabilities. `span_prob` uses inverse perplexity (geometric mean token
probability) and `token_prob` keeps per-token probabilities; both are
implemented ablations.

The optional `zero_adv_inject_beta` is useful for zero-variance policy-gradient
groups. `span_norm_ref_tokens` bounds the extra reversal mass for unusually
long offending spans while preserving the base sign flip.

## Usage

```bash
python -m pip install -e '.[test]'
pytest
python scripts/check_public_snapshot.py
python examples/apply_c3.py
```

Integrate `apply_c3` after GAE and any ordinary advantage normalization, before
the actor loss. Do not feed its output into the critic target. Environment
detectors and dataset-specific prompt logic belong in the caller's repository;
keeping them out of this package is part of the public/private boundary.

## Releasing a public snapshot

Run the snapshot scanner and inspect `git diff --stat` before publishing. Do not
add `.env` files, access tokens, service endpoints, absolute workstation or
mounted-storage paths, model/checkpoint directories, raw rollout dumps, or
private experiment reports. Use the placeholders in the config and README and
pass deployment-specific values through the downstream launcher's environment.

## License

Apache-2.0. See the upstream project license terms before combining this code
with third-party trainer code.
