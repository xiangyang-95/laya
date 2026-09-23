import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch

from laya.agent import (
    _device_available,
    _empty_accelerator_cache,
    _resolve_device,
    _should_fallback_to_cpu,
    _supports_amp,
)


class XpuSupportTests(unittest.TestCase):
    def test_xpu_availability_uses_torch_xpu(self):
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            self.skipTest("this PyTorch build has no XPU API")
        with patch.object(xpu, "is_available", return_value=True):
            self.assertTrue(_device_available("xpu"))

    def test_auto_device_selects_xpu_when_cuda_is_unavailable(self):
        with patch("laya.agent._device_available", side_effect=lambda kind: kind == "xpu"):
            self.assertEqual(_resolve_device(None), torch.device("xpu"))

    def test_explicit_xpu_device_is_used_when_available(self):
        with patch("laya.agent._device_available", return_value=True):
            self.assertEqual(_resolve_device("xpu"), torch.device("xpu"))

    def test_explicit_cpu_device_remains_available(self):
        self.assertEqual(_resolve_device("cpu"), torch.device("cpu"))

    def test_explicit_unavailable_xpu_falls_back_to_cpu(self):
        output = StringIO()
        with patch("laya.agent._device_available", return_value=False), redirect_stdout(output):
            device = _resolve_device("xpu")
        self.assertEqual(device, torch.device("cpu"))
        self.assertIn("Intel GPU (XPU) requested", output.getvalue())

    def test_amp_is_enabled_for_xpu(self):
        self.assertTrue(_supports_amp("xpu"))
        self.assertFalse(_supports_amp("cpu"))
        self.assertFalse(_supports_amp("mps"))

    def test_xpu_memory_errors_fall_back_to_cpu(self):
        self.assertTrue(_should_fallback_to_cpu(RuntimeError("XPU out of memory"), "xpu"))
        self.assertFalse(_should_fallback_to_cpu(RuntimeError("unrelated error"), "xpu"))
        self.assertFalse(_should_fallback_to_cpu(RuntimeError("out of memory"), "cpu"))

    def test_accelerator_cache_cleanup_includes_xpu(self):
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            self.skipTest("this PyTorch build has no XPU API")
        with patch.object(torch.cuda, "is_available", return_value=False), \
                patch.object(xpu, "is_available", return_value=True), \
                patch.object(xpu, "empty_cache") as empty_cache:
            _empty_accelerator_cache()
        empty_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()