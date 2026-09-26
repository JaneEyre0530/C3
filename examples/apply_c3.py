"""Minimal adapter example; feed real events from your own verifier."""

import torch

from c3 import apply_c3, policy_confidence


advantages = torch.tensor([0.4, 0.2, -0.1, 0.3], dtype=torch.float32)
loss_mask = torch.ones(4)
events = [{"pattern": "invalid_action", "value": -1.0, "target": {"start": 1, "end": 3}}]
log_probs = torch.log(torch.tensor([0.8, 0.5, 0.7, 0.9]))
confidence = policy_confidence(log_probs, loss_mask, events, mode="span_prob")
result = apply_c3(advantages, loss_mask, events, confidence=confidence)
print(result.advantages.tolist())
print(result.metrics)
