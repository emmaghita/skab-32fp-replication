# Quantized version of my TinyTransformer (model.py), for QAT.
# Same architecture as Fig. 2, but every layer comes from ElasticAI.Creator,
# so it can later be turned into VHDL.
# Replication of Ling et al.: follows their models/quant/QuantTransformer.py.
# Differences from model.py: the library's attention has no dropout (model.py: 0.2).

import torch
import torch.nn as nn

from elasticai.creator.nn.integer.linear import Linear
from elasticai.creator.nn.integer.addition import Addition
from elasticai.creator.nn.integer.encoder import Encoder
from elasticai.creator.nn.integer.avgpooling1d import AVGPooling1d

from core.model import PositionalEncoding  # my own PE from the FP32 model (fixed, never trained)


class QuantTinyTransformer(nn.Module):
    def __init__(self, num_in, num_out, d_model=24, window=24, quant_bits=6, device="cpu"):
        super().__init__()
        self.config = dict(num_in=num_in, num_out=num_out, d_model=d_model, window=window,
                           quant_bits=quant_bits)          # to rebuild it (checkpoints, to_integer)
        self.quant_bits = quant_bits
        # int_mode = False: QAT, floats snapped to the b-bit grid (training + accuracy)
        # int_mode = True:  integer-only, the same arithmetic as the FPGA (use to_integer())
        self.int_mode = False

        # positional encoding: computed once, stored as a fixed tensor [1, window, d_model]
        pe = PositionalEncoding(d_model, window).pe            # [window, d_model]
        self.register_buffer("pos_info", pe.unsqueeze(0))

        common = dict(quant_bits=quant_bits, device=device)   # every layer gets these

        # the same 5 blocks as model.py; the names become VHDL file names later
        self.input_linear = Linear(
            name="input_linear", in_features=num_in, out_features=d_model,
            num_dimensions=window, bias=True, **common)
        self.pos_info_addition = Addition(                     # model.py: x + pe
            name="add_pos_info", num_features=d_model, num_dimensions=window, **common)
        self.encoder = Encoder(                                # model.py: EncoderLayer
            name="encoder", window_size=window, d_model=d_model, ffn_dim=4 * d_model,
            nhead=1, num_enc_layers=1, do_int_forward=False, **common)
        # the library divides by in_features in precompute() ("e.g., seq_len"),
        # so in_features must be the window length, not d_model
        self.avg_pooling = AVGPooling1d(                       # model.py: x.mean(dim=1)
            name="average_pooling", in_features=window, out_features=d_model,
            in_num_dimensions=window, out_num_dimensions=1, **common)
        self.output_linear = Linear(
            name="output_linear", in_features=d_model, out_features=num_out,
            num_dimensions=1, bias=True, **common)

    def forward(self, x):                                      # x: [B, 24, 8]
        if self.int_mode:
            return self.int_forward(x)

        # QAT path. Rule: each layer's input quantizer = the previous layer's output
        # quantizer, so the grids match and no re-quantizing is needed between layers.
        # Only two tensors get their own quantizer: the model input and the PE.
        x = self.input_linear(x, given_inputs_QParams=None)
        x = self.pos_info_addition(
            inputs1=x, inputs2=self.pos_info,
            given_inputs1_QParams=self.input_linear.outputs_QParams,
            given_inputs2_QParams=None)
        x = self.encoder(x, given_inputs_QParams=self.pos_info_addition.outputs_QParams)
        x = self.avg_pooling(x, given_inputs_QParams=self.encoder.outputs_QParams)
        x = self.output_linear(x, given_inputs_QParams=self.avg_pooling.outputs_QParams)
        return x.squeeze(1)                                    # [B, 1, out] -> [B, out], like model.py

    def precompute(self):
        # turn the trained model into integer constants:
        # integer weights/biases, multiply-and-shift factors, softmax tables, quantized PE.
        # Changes quantizer state for good (e.g. bias bit width), so call it once, after training.
        self.input_linear.precompute()
        self.q_pos_info = self.pos_info_addition.inputs2_QParams.quantize(self.pos_info)
        self.pos_info_addition.precompute()
        self.encoder.sequential.precompute()
        self.avg_pooling.precompute()
        self.output_linear.precompute()

    @torch.no_grad()
    def int_forward(self, x):
        assert not self.training, "integer mode only works in eval mode"
        assert hasattr(self, "q_pos_info"), "call precompute() first (or use to_integer())"

        q = self.input_linear.inputs_QParams.quantize(x).to("cpu")   # door: float -> int
        q = self.input_linear.int_forward(q_inputs=q)                 # integers only from here
        q = self.pos_info_addition.int_forward(q_inputs1=q, q_inputs2=self.q_pos_info)
        q = self.encoder.int_forward(q_inputs=q)
        q = self.avg_pooling.int_forward(q_inputs=q)
        q = self.output_linear.int_forward(q_inputs=q)
        y = self.output_linear.outputs_QParams.dequantize(q)         # exit: int -> float
        return y.squeeze(1)


def to_integer(model):
    # integer-only copy of a trained QAT model; the original stays trainable.
    # (deepcopy fails on the library layers, so rebuild and load the state instead).
    # clone(): the library's QParams loading shares tensors instead of copying them, so
    # without it precompute() would overwrite the original model's quantizers too
    m = QuantTinyTransformer(**model.config)
    m.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})
    m.eval()
    m.precompute()
    m.int_mode = True
    return m


if __name__ == "__main__":
    torch.manual_seed(0)
    m = QuantTinyTransformer(num_in=8, num_out=1, d_model=24, quant_bits=6)
    x, y = torch.randn(256, 24, 8), torch.randn(256, 1)

    # 1) QAT mode. Train a little first: the observers learn the ranges here, and at init
    #    BatchNorm's gamma is all 1.0, which gives a degenerate range (encoder outputs 0).
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    m.train()
    for _ in range(50):
        loss = ((m(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print("QAT output shape:", m(x).shape)                       # expect [256, 1], like model.py
    print("gradients flow:", m.input_linear.weight.grad is not None)
    print("parameters:", sum(p.numel() for p in m.parameters()))  # FP32 model.py: 7465

    # 2) integer mode on a copy, then compare with the simulated (QAT) output
    m.eval()
    with torch.no_grad():
        y_sim = m(x)
        y_int = to_integer(m)(x)
    step = m.output_linear.outputs_QParams.scale_factor.item()
    print("distinct outputs:", y_sim.unique().numel(), "(1 would mean the model collapsed)")
    print(f"simulated vs integer: max difference {(y_sim - y_int).abs().max().item() / step:.0f} "
          f"quant steps, {(y_sim != y_int).float().mean().item():.0%} of outputs differ")
