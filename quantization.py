import torch
import torch.nn as nn

def apply_quantization(model, calib_loader=None):
    """Applies dynamic quantization to model on CPU, quantizing only Linear layers."""
    m = model.cpu().eval() # move to CPU to avoid issues
    torch.backends.quantized.engine = 'qnnpack'
    qmodel = torch.quantization.quantize_dynamic(
        m,
        {nn.Linear},
        dtype=torch.qint8
    )
    return qmodel