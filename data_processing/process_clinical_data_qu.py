import argparse
import os
from typing import Iterable

import pandas as pd


DEFAULT_OUTPUT_PT = "data/processed/clinical.pt"
DEFAULT_MODEL_PATH = "pretrained_models/medgemma-1.5-4b-it"
DEFAULT_SAMPLE_COL = "样本编号"
DEFAULT_GROUP_COL = "分组"
DEFAULT_SEX_COL = "性别"
DEFAULT_AGE_COL = "年龄"
DEFAULT_STAGE_COL = "TNM 分期"


def clean_value(value, default: str = "Unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a", "unknown", "unknow"}:
        return default
    return text


def normalize_cancer_type(value) -> str:
    text = clean_value(value, default="").strip()
    if "鳞癌" in text:
        return "LUSC"
    if text in {"肺癌", "腺癌"} or "腺癌" in text:
        return "LUAD"
    return "unknow"


def build_clinical_prompt(
    row: pd.Series,
    group_col: str = DEFAULT_GROUP_COL,
    sex_col: str = DEFAULT_SEX_COL,
    age_col: str = DEFAULT_AGE_COL,
    stage_col: str = DEFAULT_STAGE_COL,
) -> str:
    cancer_type = normalize_cancer_type(row.get(group_col))
    sex = clean_value(row.get(sex_col))
    age = clean_value(row.get(age_col))
    tnm_stage = clean_value(row.get(stage_col))

    return f"""
            You are an expert clinical data assistant. Transform the following clinical characteristics into a standardized, fluent clinical summary paragraph.

            IMPORTANT RULES:
            1. DO NOT include any Patient ID or generic subject identifiers in the summary. Focus only on the clinical profile.
            2. If a value is 'Unknown', explicitly state that the information is not provided or not recorded. 
            3. DO NOT hallucinate, guess, or invent any clinical values.
            4. Output ONLY the summary paragraph, no extra conversational text.

            Clinical Characteristics:
            - Cancer Subtype: {cancer_type}
            - Age: {age}
            - Sex: {sex}
            - AJCC Pathologic Stage: {tnm_stage}
""".strip()


def prepare_clinical_dataframe(
    df: pd.DataFrame,
    sample_col: str = DEFAULT_SAMPLE_COL,
    include_only: bool = False,
) -> pd.DataFrame:
    if sample_col not in df.columns:
        raise ValueError(f"Missing sample column: {sample_col}")

    prepared = df.copy()
    prepared = prepared[prepared[sample_col].notna()].copy()
    prepared[sample_col] = prepared[sample_col].astype(str).str.strip()
    prepared = prepared[prepared[sample_col] != ""].copy()

    if include_only:
        if "include" not in prepared.columns:
            raise ValueError("--include-only was set, but the input has no 'include' column")
        prepared = prepared[pd.to_numeric(prepared["include"], errors="coerce").fillna(0).astype(int) == 1]

    prepared = prepared.drop_duplicates(subset=[sample_col], keep="first").reset_index(drop=True)
    return prepared


def apply_chat_template(tokenizer, prompts: Iterable[str]) -> list[str]:
    texts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
        texts.append(text)
    return texts


def generate_summaries(
    df: pd.DataFrame,
    tokenizer,
    model,
    batch_size: int,
    max_new_tokens: int,
    group_col: str = DEFAULT_GROUP_COL,
    sex_col: str = DEFAULT_SEX_COL,
    age_col: str = DEFAULT_AGE_COL,
    stage_col: str = DEFAULT_STAGE_COL,
) -> list[str]:
    import torch
    from tqdm import tqdm

    prompts = [
        build_clinical_prompt(row, group_col, sex_col, age_col, stage_col)
        for _, row in df.iterrows()
    ]
    summaries: list[str] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="Processing clinical batches"):
        batch_prompts = prompts[start : start + batch_size]
        texts = apply_chat_template(tokenizer, batch_prompts)
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=True,
            )

        input_length = inputs["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=True)
        summaries.extend(text.strip() for text in decoded)

    return summaries


def tokenize_summaries(sample_ids: list[str], summaries: list[str], tokenizer, max_length: int):
    import torch
    from tqdm import tqdm

    tokenizer.padding_side = "right"
    output_dict = {}

    for sample_id, text in tqdm(zip(sample_ids, summaries), total=len(sample_ids), desc="Tokenizing"):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        input_ids = tokens["input_ids"].squeeze(0).cpu()
        attention_mask = tokens["attention_mask"].squeeze(0).cpu()
        token_type_ids = tokens.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        else:
            token_type_ids = token_type_ids.squeeze(0).cpu()

        output_dict[sample_id] = {
            "text": text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    return output_dict


def read_table(path: str, sep: str | None = None) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if path.lower().endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(path, sep=sep)


def process_clinical_to_pt(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    model_path = args.model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH
    df = read_table(args.input, args.sep)
    df = prepare_clinical_dataframe(df, sample_col=args.sample_col, include_only=args.include_only)
    sample_ids = df[args.sample_col].tolist()

    print(f"Loading tokenizer and model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    summaries = generate_summaries(
        df,
        tokenizer,
        model,
        args.batch_size,
        args.max_new_tokens,
        group_col=args.group_col,
        sex_col=args.sex_col,
        age_col=args.age_col,
        stage_col=args.stage_col,
    )
    output_dict = tokenize_summaries(sample_ids, summaries, tokenizer, args.max_length)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(output_dict, args.output)
    print(f"Saved {len(output_dict)} clinical prompt records to: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert QU pathology/clinical CSV into a tokenized .pt dictionary using the MedGemma prompt workflow."
    )
    parser.add_argument("--input", required=True, help="Path to the cohort clinical CSV/TSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PT, help=f"Output .pt path. Default: {DEFAULT_OUTPUT_PT}")
    parser.add_argument(
        "--model-path",
        default=None,
        help=f"Local MedGemma model path. Default: MODEL_PATH or {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument("--sample-col", default=DEFAULT_SAMPLE_COL, help=f"Sample ID column. Default: {DEFAULT_SAMPLE_COL}")
    parser.add_argument("--group-col", default=DEFAULT_GROUP_COL, help=f"Cancer group column. Default: {DEFAULT_GROUP_COL}")
    parser.add_argument("--sex-col", default=DEFAULT_SEX_COL, help=f"Sex column. Default: {DEFAULT_SEX_COL}")
    parser.add_argument("--age-col", default=DEFAULT_AGE_COL, help=f"Age column. Default: {DEFAULT_AGE_COL}")
    parser.add_argument("--stage-col", default=DEFAULT_STAGE_COL, help=f"TNM stage column. Default: {DEFAULT_STAGE_COL}")
    parser.add_argument("--sep", default=None, help="Input separator. Default: infer from file extension")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenized max_length")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="Generated clinical summary token budget")
    parser.add_argument("--include-only", action="store_true", help="Keep only rows with include == 1")
    return parser.parse_args()


if __name__ == "__main__":
    process_clinical_to_pt(parse_args())
