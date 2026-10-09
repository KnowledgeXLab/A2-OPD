import os
import json
import torch
import numpy as np


def _to_serializable(obj):
    """把各种对象尽量转成可保存/可打印的形式"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def extract_single_sample_from_dataproto(batch, idx=0):
    """
    从 DataProto 中抽取第 idx 条样本，返回普通 dict。
    假设：
      - batch.batch 中的 tensor 第一维是 batch 维
      - batch.non_tensor_batch 中很多字段也可按 idx 取一条
    """
    sample = {
        "tensor_batch": {},
        "non_tensor_batch": {},
        "meta_info": {},
    }

    # 1) tensor 字段
    for k, v in batch.batch.items():
        try:
            if isinstance(v, torch.Tensor):
                # 取单条样本
                sample["tensor_batch"][k] = v[idx].detach().cpu()
            else:
                sample["tensor_batch"][k] = _to_serializable(v)
        except Exception as e:
            sample["tensor_batch"][k] = f"<extract failed: {e}>"

    # 2) non-tensor 字段
    if hasattr(batch, "non_tensor_batch") and batch.non_tensor_batch is not None:
        for k, v in batch.non_tensor_batch.items():
            try:
                if isinstance(v, np.ndarray):
                    sample["non_tensor_batch"][k] = v[idx]
                elif isinstance(v, (list, tuple)):
                    sample["non_tensor_batch"][k] = v[idx]
                else:
                    sample["non_tensor_batch"][k] = _to_serializable(v)
            except Exception as e:
                sample["non_tensor_batch"][k] = f"<extract failed: {e}>"

    # 3) meta_info
    if hasattr(batch, "meta_info") and batch.meta_info is not None:
        for k, v in batch.meta_info.items():
            try:
                sample["meta_info"][k] = _to_serializable(v)
            except Exception as e:
                sample["meta_info"][k] = f"<extract failed: {e}>"

    return sample


def save_dataproto_single_sample(batch, save_path, idx=0):
    """
    保存 DataProto 中的一条样本到本地 .pt 文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sample = extract_single_sample_from_dataproto(batch, idx=idx)
    torch.save(sample, save_path)
    print(f"[DEBUG] saved sample idx={idx} to {save_path}")


def save_dataproto_single_sample_with_json_preview(batch, save_path, idx=0, json_path=None):
    """
    除了保存 .pt，也可选保存一个 json 预览文件，方便直接看字段
    """
    sample = extract_single_sample_from_dataproto(batch, idx=idx)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(sample, save_path)

    if json_path is not None:
        preview = {
            "tensor_batch": {},
            "non_tensor_batch": sample["non_tensor_batch"],
            "meta_info": sample["meta_info"],
        }
        for k, v in sample["tensor_batch"].items():
            if isinstance(v, torch.Tensor):
                preview["tensor_batch"][k] = {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "preview": v.flatten()[:20].tolist(),
                }
            else:
                preview["tensor_batch"][k] = str(v)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2, default=str)

    print(f"[DEBUG] saved sample idx={idx} to {save_path}")
    if json_path is not None:
        print(f"[DEBUG] saved json preview to {json_path}")