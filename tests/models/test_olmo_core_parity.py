"""The converted Hugging Face models reproduce the paper's OLMo-core checkpoints.

Needs the OLMo-core package of ``pretrain/OLMo-core`` and the released 400M and 1.5B checkpoints, downloaded
and converted as the README describes::

    hf download davidgonmar/depth-memory-400m-ldm --include "lr1e-3/*" --local-dir checkpoints/released/400m-ldm
    python utils/convert_from_olmo_core.py --checkpoint checkpoints/released/400m-ldm/lr1e-3 \
        --output checkpoints/converted/400m-ldm

Tokens come from the held-out FineWeb-Edu shard under ``$DATA_DIR/eval`` when it exists. The models
run in BF16, so they are compared at BF16 tolerance: the mean absolute logit difference must stay
near that between the HF model's own BF16 and FP32 forwards (about 0.02-0.03 at 400M).
A checkpoint that is not downloaded is skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
CASES = [
    ("400m-ldm", {"ldm_mode": "naive"}),
    ("400m-ldm", {"ldm_mode": "fused_recurrent"}),
    ("400m-ldm", {"ldm_mode": "chunk"}),
    ("400m-attnres-block", {"attnres_mode": "fused"}),
    ("400m-mhc", {"mhc_mode": "fused"}),
    ("1p5b-ldm", {"ldm_mode": "fused_recurrent"}),
    ("1p5b-ldm", {"ldm_mode": "chunk"}),
    ("1p5b-attnres-block", {"attnres_mode": "fused"}),
    ("1p5b-mhc", {"mhc_mode": "fused"}),
]


def _tokens(batch: int = 2, length: int = 2048) -> torch.Tensor:
    data = Path(os.environ.get("DATA_DIR", REPO / "data/fineweb-edu/tokenized")) / "eval"
    shards = sorted(data.glob("*.npy"))
    if shards:
        tokens = np.memmap(shards[0], dtype=np.uint16, mode="r")[: batch * length].astype(np.int64)
        return torch.from_numpy(tokens).view(batch, length)
    return torch.randint(2, 50000, (batch, length), generator=torch.Generator().manual_seed(0))


_REFERENCE: dict[str, torch.Tensor] = {}


def _olmo_core_logits(name: str, ids: torch.Tensor) -> torch.Tensor:
    if name not in _REFERENCE:
        sys.path.insert(0, str(REPO / "utils"))
        from convert_from_olmo_core import load_olmo_core

        model, _ = load_olmo_core(REPO / "checkpoints/released" / name / "lr1e-3")
        model = model.cuda().to(torch.bfloat16)
        # Training keeps LDM's forgetting logits and mHC's controllers FP32.
        for attribute in ("ldm_retention", "mhc_parameter_bank"):
            if getattr(model, attribute, None) is not None:
                getattr(model, attribute).float()
        for module in model.modules():
            # OLMo-core's split tables cover H100 and B200; one split is upstream's projection.
            if hasattr(module, "inference_projection_splits"):
                module.inference_projection_splits = 1
        with torch.no_grad():
            _REFERENCE[name] = model(ids)[:, :-1].float()
        del model
        torch.cuda.empty_cache()
    return _REFERENCE[name]


@pytest.mark.parametrize(("name", "overrides"), CASES, ids=[f"{n}-{next(iter(o.values()))}" for n, o in CASES])
def test_olmo_core_parity(name: str, overrides: dict):
    pytest.importorskip("olmo_core")
    if not (REPO / "checkpoints/released" / name / "lr1e-3" / "config.json").exists():
        pytest.skip(f"released checkpoint {name} not downloaded")
    if not (REPO / "checkpoints/converted" / name / "config.json").exists():
        pytest.skip(f"converted checkpoint {name} missing")
    from transformers import AutoModelForCausalLM

    import depth_mixer  # noqa: F401

    ids = _tokens().cuda()
    reference = _olmo_core_logits(name, ids)
    model = AutoModelForCausalLM.from_pretrained(REPO / "checkpoints/converted" / name, dtype=torch.bfloat16, **overrides)
    with torch.no_grad():
        logits = model.cuda().eval()(ids).logits[:, :-1].float()

    labels = ids[:, 1:].flatten()
    loss = F.cross_entropy(logits.flatten(0, 1), labels)
    reference_loss = F.cross_entropy(reference.flatten(0, 1), labels)
    agreement = (logits.argmax(-1) == reference.argmax(-1)).float().mean()
    mean_difference = (logits - reference).abs().mean()
    assert abs(loss - reference_loss) < 2e-3, (loss.item(), reference_loss.item())
    assert mean_difference < 0.05, mean_difference.item()
    assert agreement > 0.98, agreement.item()
