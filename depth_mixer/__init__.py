"""Depth mixers: methods that replace the residual connection of a transformer across its layers.

The layout follows flash-linear-attention: ``ops`` holds the kernels and their naive references,
``layers`` the modules that apply them inside a block, ``models`` the Hugging Face models, and
``modules`` the shared building blocks.
"""

import importlib

__version__ = "0.1.0"

__all__: list[str] = []


def _export_public_api(module_name: str) -> None:
    module = importlib.import_module(module_name)
    globals()[module_name.rsplit(".", maxsplit=1)[-1]] = module
    for name in module.__all__:
        if name.endswith("Config"):
            continue
        globals()[name] = getattr(module, name)
        __all__.append(name)


_export_public_api("depth_mixer.layers")
_export_public_api("depth_mixer.models")

del _export_public_api
