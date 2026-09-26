# Extraction map

This repository is the sanitized algorithm boundary extracted from the private
trainer integration. The mapping is intentionally descriptive; the public
package does not import or require the private project.

| Private integration responsibility | Public module |
| --- | --- |
| Build bad events from parsed Search-R1 turns | `c3/events.py:detect_events` |
| Aggregate detached token/span policy confidence | `c3/core.py:policy_confidence` |
| Apply actor-only verifier correction after GAE/whitening | `c3/core.py:apply_c3` |
| Paper defaults and event severities | `configs/c3_public.toml` |

The public boundary stops at serialized event spans. Prompt templates, search
backends, WebShop/AppWorld services, model/tokenizer paths, distributed
launchers, checkpoints, raw datasets, and experiment telemetry remain outside
this repository and must be supplied by a downstream integration.

The contract preserved by the extraction is:

1. Environment rewards, GAE returns, and critic targets are not changed.
2. Only positive environment advantages on detector-covered tokens are
   reversed by default.
3. Overlapping events use the maximum clipped severity per token.
4. Event values, token offsets, and policy confidence are detached inputs to
   the actor-side transformation.
