"""Compatibility shim for pytorchvideo on torchvision >= 0.17.

``torchvision.transforms.functional_tensor`` was removed in torchvision 0.17, but
released pytorchvideo (0.1.5) still imports it. Alias it to the public
``torchvision.transforms.functional`` before pytorchvideo is imported.
"""

import sys

import torchvision


def patch_torchvision_for_pytorchvideo():
    try:
        from torchvision.transforms import functional_tensor  # noqa: F401
    except ImportError:
        sys.modules["torchvision.transforms.functional_tensor"] = torchvision.transforms.functional
