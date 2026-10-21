"""lm-evaluation-harness entry point for depth-mixer models.

    python -m evals.harness --model depth_mixer \
        --model_args pretrained=checkpoints/converted/400m-ldm,dtype=bfloat16 \
        --tasks arc_easy,arc_challenge,hellaswag,piqa,winogrande,openbookqa,lambada_openai,wikitext \
        --batch_size 32 --device cuda:0

These are the paper's zero-shot tasks. A model's kernel mode can be overridden through its config
fields, e.g. ``--model_args pretrained=...,ldm_mode=chunk``.
"""

from __future__ import annotations

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

import depth_mixer  # noqa: F401  (registers the models with the Auto classes)


@register_model('depth_mixer')
class DepthMixerLMWrapper(HFLM):
    def __init__(self, **kwargs) -> DepthMixerLMWrapper:
        # The depth-mixer configs are causal LMs that transformers' own registries do not list.
        kwargs.setdefault("backend", "causal")
        super().__init__(**kwargs)


if __name__ == "__main__":
    cli_evaluate()
