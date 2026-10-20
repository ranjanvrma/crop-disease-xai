"""INT8 post-training quantization (PT2E) and a model wrapper for the quantized graph.

The float model is exported with two outputs, logits and the final feature
maps, which is exactly what the Android app receives. The classifier weights
are kept in float alongside, so the closed-form Grad-CAM++ can be computed
from the INT8 feature maps without any gradients.
"""

import io
import warnings

import torch
from torch import nn


class TwoOutputs(nn.Module):
    """Float model -> (logits, final feature maps)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.net.forward_features(x)
        return self.net.forward_head(features), features


def _pt2e_tools():
    """XNNPACK quantizer (the scheme ExecuTorch runs on Android) if available,
    otherwise torchao's x86 quantizer. Returns (prepare, convert, quantizer, name)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # deprecation notices for torch.ao
        try:
            from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
            from torch.ao.quantization.quantizer.xnnpack_quantizer import (
                XNNPACKQuantizer,
                get_symmetric_quantization_config,
            )
            quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True))
            return prepare_pt2e, convert_pt2e, quantizer, "xnnpack_symmetric_per_channel"
        except ImportError:
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
            from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
                X86InductorQuantizer,
                get_default_x86_inductor_quantization_config,
            )
            quantizer = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config())
            return prepare_pt2e, convert_pt2e, quantizer, "x86_inductor_default"


def _export(module: nn.Module, image_size: int):
    batch = torch.export.Dim("batch", max=1024)
    example = (torch.randn(2, 3, image_size, image_size),)
    return torch.export.export(module, example, dynamic_shapes=({0: batch},))


@torch.no_grad()
def quantize_int8(model: nn.Module, calib_images: torch.Tensor, image_size: int, batch_size: int = 32) -> tuple[bytes, str]:
    """Post-training INT8 quantization on the CPU. Returns (serialized program, quantizer name)."""
    prepare, convert, quantizer, name = _pt2e_tools()
    model = model.cpu().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)  # torch.ao deprecation notices
        prepared = prepare(_export(TwoOutputs(model), image_size).module(), quantizer)
        for start in range(0, len(calib_images), batch_size):
            prepared(calib_images[start:start + batch_size])       # observers record activation ranges
        quantized = convert(prepared)
    buf = io.BytesIO()
    torch.export.save(_export(quantized, image_size), buf)
    return buf.getvalue(), name


class QuantizedModel(nn.Module):
    """The INT8 graph behind the same interface as the float models.

    ``forward`` returns logits, ``forward_features`` the final feature maps and
    ``get_classifier`` the float classifier, so evaluation, faithfulness metrics
    and closed-form Grad-CAM++ work unchanged. It has no gradients.
    """

    def __init__(self, program: bytes, classifier: nn.Linear, device: torch.device):
        super().__init__()
        # The graph uses torch.ops.quantized_decomposed.*, which only exist once their module is
        # imported; a fresh process (evaluate/explain) must register them before deserializing.
        import torch.ao.quantization.fx._decomposed  # noqa: F401
        ep = torch.export.load(io.BytesIO(program))
        self.device = torch.device("cpu")
        if device.type != "cpu":
            try:  # the quantize/dequantize reference ops are device-agnostic; fall back to CPU if not
                from torch.export.passes import move_to_device_pass
                ep = move_to_device_pass(ep, device)
                self.device = device
            except Exception as exc:  # noqa: BLE001
                print(f"INT8 model stays on CPU ({type(exc).__name__}: {exc})")
        self.graph = ep.module()
        self.classifier = classifier.to(self.device)

    def _run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, features = self.graph(x.to(self.device).contiguous())
        return logits.to(x.device), features.to(x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)[0]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)[1]

    def get_classifier(self) -> nn.Linear:
        return self.classifier

    def eval(self):  # exported graphs don't support train()/eval(); they are always in eval mode
        return self

    def to(self, *args, **kwargs):  # device placement is fixed at load time
        return self
