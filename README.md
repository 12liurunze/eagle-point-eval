# Eagle Point Eval

This repository adapts **EAGLE-EYE** speculative decoding to **PointLLM** for 3D point-cloud language generation. It provides scripts for generating draft-head training data, training a PointLLM draft head, running EAGLE-style generation, and comparing speculative decoding against vanilla autoregressive PointLLM decoding.

The project is mainly intended for PointLLM-7B experiments on Objaverse-style point-cloud data.

## Features

- PointLLM draft-head training data generation
- PointLLM EAGLE draft-head training
- Single-sample speculative decoding inference
- Batch comparison between autoregressive PointLLM and EAGLE decoding
- Timing summary for speedup evaluation
- Compatibility scripts for Linux and PowerShell

## Repository Structure

```text
.
├── EAGLE_EYE/
│   ├── eagle_eye/
│   │   ├── evaluation/
│   │   │   ├── compare_pointllm_eagle.py
│   │   │   └── gen_ee_answer_pointllm.py
│   │   ├── ge_data/
│   │   │   └── get_data_all_pointllm.py
│   │   ├── model/
│   │   │   ├── point_ee_model.py
│   │   │   └── pointllm_tree_modeling.py
│   │   └── train/
│   │       ├── pointllm_7B_config.json
│   │       └── train_pointllm.py
│   ├── requirements.txt
│   └── setup.py
└── scripts/
    ├── pointllm_generate_data.sh
    ├── pointllm_train_head.sh
    ├── pointllm_infer_one.sh
    └── pointllm_compare_eagle.sh
```

## Environment

Install the package in editable mode:

```bash
cd EAGLE_EYE
pip install -e .
```

The PointLLM codebase must also be available locally. The scripts use the following default paths on AutoDL-style servers:

```text
PointLLM repo:      /root/autodl-tmp/pointLLM
Base model:         /root/autodl-tmp/point7B_v1.1
Point cloud data:   /root/autodl-tmp/pointLLM/data/objaverse_data
Validation JSON:    /root/autodl-tmp/pointLLM/data/anno_data/PointLLM_brief_description_val_200_GT.json
Draft head output:  /root/autodl-tmp/pointllm_eagle_head
```

You can override these defaults with environment variables.

## Generate Draft-Head Training Data

```bash
bash scripts/pointllm_generate_data.sh
```

Common overrides:
model weights and dataset could refer to this link:https://github.com/12liurunze/PointLLM
```bash
POINTLLM_REPO=/path/to/pointLLM \
BASE_MODEL=/path/to/point7B_v1.1 \
POINT_CLOUD_DATA=/path/to/objaverse_data \
ANNOTATION=/path/to/PointLLM_complex_instruction_70K.json \
OUTPUT_DIR=/path/to/pointllm_eagle_data \
bash scripts/pointllm_generate_data.sh
```

Useful range controls:

```bash
START=0 END=10000 INDEX=0 bash scripts/pointllm_generate_data.sh
```

## Train the PointLLM Draft Head

```bash
bash scripts/pointllm_train_head.sh
```

Common overrides:

```bash
DATA_DIR=/path/to/pointllm_eagle_data \
HEAD_DIR=/path/to/pointllm_eagle_head \
BATCH_SIZE=24 \
NUM_EPOCHS=20 \
MIXED_PRECISION=no \
bash scripts/pointllm_train_head.sh
```

The trained draft head is saved to `HEAD_DIR`.

## Run One Inference Example

Use the first sample from the validation JSON:

```bash
VAL_INDEX=0 bash scripts/pointllm_infer_one.sh
```

Run with a specific object id and question:

```bash
AUTO_READ_OBJECT_IDS=0 \
bash scripts/pointllm_infer_one.sh <object_id> "Describe this 3D object in detail."
```

Important generation parameters:

```bash
MAX_NEW_TOKENS=256 MAX_LENGTH=2048 TORCH_DTYPE=float32 bash scripts/pointllm_infer_one.sh
```

## Compare Autoregressive and EAGLE Decoding

```bash
bash scripts/pointllm_compare_eagle.sh
```

This runs both vanilla PointLLM autoregressive decoding and EAGLE speculative decoding on the validation set, then writes:

```text
OUTPUT_JSONL:  per-sample outputs and timing
SUMMARY_JSON:  aggregate timing and speedup summary
```

Example:

```bash
HEAD_DIR=/root/autodl-tmp/pointllm_eagle_head \
VAL_JSON=/root/autodl-tmp/pointLLM/data/anno_data/PointLLM_brief_description_val_200_GT.json \
OUTPUT_JSONL=/root/autodl-tmp/pointllm_compare_eagle.jsonl \
SUMMARY_JSON=/root/autodl-tmp/pointllm_compare_eagle_summary.json \
bash scripts/pointllm_compare_eagle.sh
```

To evaluate a subset:

```bash
START=0 END=20 bash scripts/pointllm_compare_eagle.sh
```

## Main Environment Variables

| Variable | Description |
| --- | --- |
| `POINTLLM_REPO` | Path to the PointLLM repository |
| `BASE_MODEL` | Path to the original PointLLM base model |
| `POINT_CLOUD_DATA` | Directory containing point-cloud `.npy` data |
| `ANNOTATION` | Training annotation JSON |
| `VAL_JSON` | Evaluation annotation JSON |
| `DATA_DIR` | Generated draft-head training data directory |
| `HEAD_DIR` | Draft-head checkpoint directory |
| `OUTPUT_JSONL` | Per-sample evaluation output |
| `SUMMARY_JSON` | Aggregate evaluation summary |
| `TORCH_DTYPE` | Model dtype, default `float32` |
| `MAX_NEW_TOKENS` | Maximum generated tokens |
| `TEMPERATURE` | Sampling temperature, default `0.0` |
| `TOP_P` | Top-p sampling value |
| `TOP_K` | Top-k sampling value |

## Notes

- The verifier path uses the original PointLLM model.
- The draft path uses the trained EAGLE draft head.
- Keep the training data generation setup consistent with inference and evaluation settings.
- Large model weights, generated hidden states, and evaluation outputs should not be committed to Git.

## Acknowledgements

This project builds on EAGLE-EYE and PointLLM. It is intended as an experimental codebase for studying speculative decoding acceleration in point-cloud language models.
