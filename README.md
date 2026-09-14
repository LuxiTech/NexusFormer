# NexusFormer

[![Paper](https://img.shields.io/badge/arXiv-2604.19147-b31b1b.svg)](https://arxiv.org/abs/2604.19147)

PyTorch implementation of **Nexusformer: Nonlinear Attention Expansion for Stable and Inheritable Transformer Scaling**.

NexusFormer replaces the fixed linear projections used by standard Transformer attention with a three-stage **Nexus-Rank** mapping:

```text
D  --linear + GELU-->  M  --linear + GELU-->  A  --linear-->  D
```

where `D < M < A`. The two expanded nonlinear feature spaces increase representational capacity and provide two structural axes that can be enlarged while retaining pretrained weights. In the paper, zero-initialized blocks are used during expansion so that added capacity does not perturb the function represented by the original model at initialization.

The paper reports language-modeling and reasoning experiments at multiple model scales, including progressive growth from 240M to 440M parameters. See the [paper](https://arxiv.org/abs/2604.19147) for the method, scaling procedure, experimental protocol, and complete results.

## Implementation overview

This repository contains a Hugging Face-compatible causal language model implementation. The historical Python class names use the `PTransformer` prefix.

| Paper notation | Configuration/code | Role |
| --- | --- | --- |
| `D` | `hidden_size` / `input_channels` | Transformer hidden width |
| `M` | `param_token_num` / `hidden_channels` | Intermediate Nexus-Rank width |
| `A` | `vertical_channels` | Expanded Nexus-Rank width |
| `D_out` | `output_channels` | Projection output width |

The `EquivalentMLP` module implements the Nexus-Rank path. It is used for the attention `Q`, `K`, `V`, and output projections, and by the Transformer block's feed-forward path. Attention uses rotary position embeddings and FlashAttention.

## Repository layout

```text
.
├── README.md
└── ptransformer
    ├── Pattention.py                  # Nexus-Rank/Pattention layers
    ├── configuration_ptransformer.py  # Model configuration
    ├── modeling_ptransformer.py       # Transformer and causal-LM classes
    ├── pattn_token.py                 # Nexus attention implementation
    └── __init__.py                    # Public package exports
```

## Requirements

- Python 3.10 or newer
- PyTorch with CUDA support
- Hugging Face Transformers
- Flash Linear Attention (`fla` Python package)
- FlashAttention 2
- einops
- A supported accelerator; the current attention path does not include a CPU fallback

Create an isolated environment and install a PyTorch build appropriate for your CUDA driver first. Then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

# Install PyTorch from https://pytorch.org/get-started/locally/ first.
pip install transformers einops
pip install "flash-linear-attention[cuda]"
pip install flash-attn --no-build-isolation
```

The Flash Linear Attention project supports several accelerator backends, but this repository's `pattn_token.py` currently calls the CUDA-oriented `flash_attn` API directly. The exact PyTorch, CUDA, and FlashAttention versions must therefore be compatible with one another.

## Quick start

Run the example from the repository root so that the local `ptransformer` package is importable:

```python
import torch

from ptransformer import PTransformerConfig, PTransformerForCausalLM


config = PTransformerConfig(
    vocab_size=32_000,
    hidden_size=256,          # D
    num_hidden_layers=4,
    num_heads=8,
    num_kv_heads=8,
    intermediate_size=768,
    param_token_num=384,      # M
    vertical_channels=512,    # A
    max_position_embeddings=2_048,
    fuse_norm=True,
    fuse_cross_entropy=True,
)

device = torch.device("cuda")
model = PTransformerForCausalLM(config).to(device=device, dtype=torch.bfloat16)
input_ids = torch.randint(0, config.vocab_size, (1, 128), device=device)

with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=False)

print(outputs.logits.shape)  # torch.Size([1, 128, 32000])
```

For a training step, pass `labels=input_ids`. The implementation shifts the labels internally for next-token prediction:

```python
model.train()
outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
outputs.loss.backward()
```

## Main configuration fields

| Field | Default | Description |
| --- | ---: | --- |
| `hidden_size` | `2048` | Model width (`D`) |
| `num_hidden_layers` | `24` | Number of Transformer blocks |
| `num_heads` | `32` | Number of query heads |
| `num_kv_heads` | `None` | Number of key/value heads; defaults to `num_heads` |
| `param_token_num` | `1024` | First Nexus-Rank expansion width (`M`) |
| `vertical_channels` | `768` | Second Nexus-Rank expansion width (`A`) |
| `intermediate_size` | computed | Feed-forward expanded width |
| `max_position_embeddings` | `2048` | Maximum rotary-position cache length |
| `window_size` | `None` | Optional causal sliding-attention window |
| `fuse_norm` | `True` | Use the fused FLA RMSNorm implementation |
| `fuse_cross_entropy` | `True` | Use fused FLA cross-entropy during training |

`hidden_size` must be divisible by `num_heads`, and `num_heads` must be divisible by `num_kv_heads`. For the intended Nexus-Rank design, choose widths satisfying `hidden_size < param_token_num < vertical_channels`.

## Current scope

- The repository contains the model architecture, not pretrained checkpoints, tokenizer files, datasets, training orchestration, or evaluation scripts.
- `AutoConfig`, `AutoModel`, and `AutoModelForCausalLM` registration calls are currently disabled in `ptransformer/__init__.py`; import the classes directly as shown above.
- `output_attentions=True` is not currently supported.
- FlashAttention is required by the active attention implementation.
- The progressive zero-initialized model-growth procedure described in the paper is not exposed as a standalone utility in this source snapshot.

## Citation

If you use NexusFormer in your work, please cite:

```bibtex
@misc{zhao2026nexusformer,
  title         = {Nexusformer: Nonlinear Attention Expansion for Stable and Inheritable Transformer Scaling},
  author        = {Weijie Zhao and Mingquan Liu and Bolun Wang and Simo Wu and Nuobei Xie and Rui-Jie Zhu and Peng Zhou},
  year          = {2026},
  eprint        = {2604.19147},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2604.19147},
  url           = {https://arxiv.org/abs/2604.19147}
}
```

## License

No project-level license file is included in this source snapshot. Until the maintainers add one, use, redistribution, and modification rights are not granted by this repository. Individual source files may contain their own attribution notices.
