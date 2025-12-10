# merge the delta on base model to instruct model by the Shadow-FT scheme
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from transformers import AutoTokenizer, AutoModelForCausalLM

def merge_models_add_diff(modelA, modelB, modelC):
    """
    对三个模型逐层执行： C += (B - A)
    要求三个模型结构一致（同名参数 shape 完全一致）
    """

    stateA = modelA.state_dict()
    stateB = modelB.state_dict()
    stateC = modelC.state_dict()

    merged = {}

    for name in stateA.keys():
        pA = stateA[name]
        pB = stateB[name]
        pC = stateC[name]

        # 只处理 Tensor
        if not torch.is_tensor(pA):
            merged[name] = pC
            continue

        # shape 检查
        if pA.shape != pB.shape or pA.shape != pC.shape:
            raise ValueError(f"Shape mismatch at {name}: "
                             f"A{pA.shape}, B{pB.shape}, C{pC.shape}")

        # diff = B - A，然后 C = C + diff
        merged[name] = pC + (pB - pA)

    # 加载新的权重
    modelA.load_state_dict(merged)
    return modelA

model_path = "../models/PeoplesDaily-Qwen3-4B-Base"
base_model_path = "Qwen/Qwen3-4B-Base"
instruct_model_path = "Qwen/Qwen3-4B-Instruct-2507"
merged_path = "../models/PeoplesDaily-Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
instruct_model = AutoModelForCausalLM.from_pretrained(instruct_model_path, trust_remote_code=True)

sft_model = merge_models_add_diff(base_model, model, instruct_model)

sft_model = sft_model.to(dtype=torch.bfloat16)
sft_model.save_pretrained(
    save_directory=merged_path,
    safe_serialization=True,          # ensures .safetensors, not .bin
    max_shard_size="4.5GB"              # splits into ~4 GB shards
)


