from __future__ import annotations

import sys
import types


def _install_torchvision_functional_tensor_shim() -> None:
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        from torchvision.transforms.functional_tensor import rgb_to_grayscale  # noqa: F401
        return
    except ImportError:
        from torchvision.transforms.functional import rgb_to_grayscale

        shim = types.ModuleType(module_name)
        shim.rgb_to_grayscale = rgb_to_grayscale
        sys.modules[module_name] = shim


_install_torchvision_functional_tensor_shim()
