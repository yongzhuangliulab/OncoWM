"""Inspect the structure of a trusted PyTorch .pt file without changing it."""

import argparse
from collections.abc import Mapping
from typing import Any

import torch


def describe(value: Any, depth: int = 0) -> str:
    if isinstance(value, torch.Tensor):
        finite = ""
        if value.is_floating_point():
            nan_count = int(torch.isnan(value).sum().item())
            inf_count = int(torch.isinf(value).sum().item())
            finite = f", nan={nan_count}, inf={inf_count}"
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}{finite})"
    if isinstance(value, Mapping):
        keys = list(value.keys())
        return f"dict(keys={keys})" if depth else f"dict(len={len(value)})"
    if isinstance(value, str):
        preview = value.replace("\n", " ")[:80]
        return f"str(len={len(value)}, preview={preview!r})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return f"{type(value).__name__}({value!r})"


def inspect_pt(path: str, max_items: int) -> None:
    value = torch.load(path, map_location="cpu", weights_only=True)
    print(f"File: {path}")
    print(f"Top-level type: {type(value).__name__}")
    if isinstance(value, torch.Tensor):
        print(describe(value))
        return
    if isinstance(value, Mapping):
        print(f"Records: {len(value)}")
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                if len(value) > max_items:
                    print(f"... {len(value) - max_items} more record(s)")
                break
            print(f"[{index}] key={key!r} value={describe(item, depth=1)}")
            if isinstance(item, Mapping):
                for subkey, subvalue in item.items():
                    print(f"    {subkey!r}: {describe(subvalue, depth=2)}")
        return
    print(describe(value))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display keys, shapes, dtypes, NaN and Inf counts in a trusted PT file."
    )
    parser.add_argument("path", help="Path to a trusted .pt file")
    parser.add_argument("--max-items", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inspect_pt(args.path, max(0, args.max_items))
