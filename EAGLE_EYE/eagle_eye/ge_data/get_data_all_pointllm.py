import argparse
import copy
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoConfig


def add_pointllm_to_path(pointllm_repo_path):
    if pointllm_repo_path and pointllm_repo_path not in sys.path:
        sys.path.insert(0, pointllm_repo_path)


def load_checkpoint_tensors_by_prefix(model, checkpoint_dir, prefixes):
    wanted = set(model.state_dict().keys())
    patch = {}

    single_bin = os.path.join(checkpoint_dir, "pytorch_model.bin")
    bin_files = []
    if os.path.exists(single_bin):
        bin_files.append(single_bin)
    else:
        bin_files.extend(
            os.path.join(checkpoint_dir, name)
            for name in sorted(os.listdir(checkpoint_dir))
            if name.endswith(".bin")
        )

    for bin_file in bin_files:
        weights = torch.load(bin_file, map_location="cpu")
        for key, value in weights.items():
            if key in wanted and any(key.startswith(prefix) for prefix in prefixes):
                patch[key] = value

    if patch:
        model.load_state_dict(patch, strict=False)
        print(f"[INFO] Manually loaded {len(patch)} tensors from checkpoint shards: {sorted(patch)}")


def build_dataset(args, tokenizer, point_backbone_config):
    from pointllm.data.object_point_dataset import ObjectPointCloudDataset
    from pointllm import conversation as conversation_lib

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    class DataArgs:
        pass

    data_args = DataArgs()
    data_args.point_backbone_config = point_backbone_config
    data_args.data_debug_num = 0
    data_args.split_train_val = False
    data_args.split_ratio = 1.0

    dataset = ObjectPointCloudDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        tokenizer=tokenizer,
        pointnum=args.pointnum,
        split="train",
        conversation_types=tuple(args.conversation_types.split(",")),
        use_color=True,
        data_args=data_args,
    )
    end = min(args.end, len(dataset)) if args.end > 0 else len(dataset)
    return dataset, range(args.start, end)


def make_loss_mask(labels):
    return (labels != -100).long()


@torch.no_grad()
def build_training_item(model, sample, device, dtype):
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    point_clouds = sample["point_clouds"].unsqueeze(0).to(device=device, dtype=dtype)
    loss_mask = make_loss_mask(sample["labels"]).to(input_ids.device)

    outputs = model(
        input_ids=input_ids,
        point_clouds=point_clouds,
        output_hidden_states=True,
    )

    return {
        "input_ids": input_ids.cpu()[0],
        "inputs_embeds": outputs.hidden_states[0].cpu()[0],
        "hidden_state": outputs.hidden_states[-1].cpu()[0],
        "loss_mask": loss_mask.cpu(),
    }


def write_data(outdir, data_point):
    os.makedirs(outdir, exist_ok=True)
    idx = len([name for name in os.listdir(outdir) if name.endswith(".ckpt")])
    torch.save(data_point, os.path.join(outdir, f"data_{idx}.ckpt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--pointllm-repo-path", default=None)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--conversation-types", default="simple_description")
    parser.add_argument("--conv-mode", default="vicuna_v1_1")
    parser.add_argument("--point-backbone-config-name", default=None)
    parser.add_argument("--force-single-point-proj", action="store_true")
    parser.add_argument("--torch-dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_mapping[args.torch_dtype]

    add_pointllm_to_path(args.pointllm_repo_path)
    from pointllm.model import PointLLMLlamaForCausalLM
    from pointllm.utils import disable_torch_init
    import pointllm.model.pointllm as pointllm_modeling

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    config = AutoConfig.from_pretrained(args.base_model_path)
    if args.point_backbone_config_name:
        config.point_backbone_config_name = args.point_backbone_config_name
    if args.force_single_point_proj:
        config.force_single_point_proj = True

        original_cfg_from_yaml_file = pointllm_modeling.cfg_from_yaml_file

        def cfg_from_yaml_file_single_proj(*cfg_args, **cfg_kwargs):
            point_bert_config = original_cfg_from_yaml_file(*cfg_args, **cfg_kwargs)
            point_bert_config.model.projection_hidden_layer = 0
            if "projection_hidden_dim" in point_bert_config.model:
                del point_bert_config.model.projection_hidden_dim
            return point_bert_config

        pointllm_modeling.cfg_from_yaml_file = cfg_from_yaml_file_single_proj

    model = PointLLMLlamaForCausalLM.from_pretrained(
        args.base_model_path,
        config=config,
        low_cpu_mem_usage=False,
        torch_dtype=dtype,
    )
    load_checkpoint_tensors_by_prefix(model, args.base_model_path, prefixes=("model.point_proj.",))
    model = model.cuda()
    model.config.use_cache = True
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()

    dataset, indices = build_dataset(args, tokenizer, model.get_model().point_backbone_config)
    outdir = os.path.join(args.outdir, str(args.index))
    for idx in tqdm(indices):
        try:
            item = build_training_item(model, copy.deepcopy(dataset[idx]), model.device, dtype)
            write_data(outdir, item)
        except Exception as exc:
            print(f"[WARN] skip index={idx}: {type(exc).__name__}: {repr(exc)}")


if __name__ == "__main__":
    main()
