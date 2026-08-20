import argparse
import os
from pathlib import Path

import pandas as pd


DEFAULT_LIST_DIR = "list"


def load_gene_list(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        genes = [line.strip() for line in handle if line.strip()]
    if not genes:
        raise ValueError(f"Gene list is empty: {path}")
    return genes


def normalize_cancer_type_arg(cancer_type: str) -> str:
    cancer = cancer_type.strip().upper()
    if cancer.startswith("TCGA-"):
        cancer = cancer[len("TCGA-") :]
    return cancer


def resolve_gene_list_path(
    gene_list: str | None,
    cancer_type: str | None,
    list_dir: str = DEFAULT_LIST_DIR,
) -> str:
    if gene_list:
        return gene_list
    if not cancer_type:
        raise ValueError("Please provide either --gene-list or --cancer-type")

    cancer = normalize_cancer_type_arg(cancer_type)
    clean_list_dir = list_dir.rstrip("/\\")
    return f"{clean_list_dir}/TCGA-{cancer}_gene_list.txt"


def build_mutation_vectors(
    df: pd.DataFrame,
    gene_list: list[str],
    sample_col: str = "sample",
) -> dict[str, list[float]]:
    if sample_col not in df.columns:
        raise ValueError(f"Missing sample column: {sample_col}")

    gene_index = {gene: idx for idx, gene in enumerate(gene_list)}
    mut_columns = [col for col in df.columns if col.startswith("mut_")]
    vectors: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        if pd.isna(row[sample_col]):
            continue
        sample_id = str(row[sample_col]).strip()
        if not sample_id:
            continue

        vector = vectors.setdefault(sample_id, [0.0] * len(gene_list))
        for col in mut_columns:
            gene = col.removeprefix("mut_")
            if gene not in gene_index:
                continue
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(value) and float(value) != 0.0:
                vector[gene_index[gene]] = 1.0

    return vectors


def default_output_path(gene_list_path: str, output_dir: str) -> str:
    stem = Path(gene_list_path).name
    if stem.endswith("_gene_list.txt"):
        stem = stem[: -len("_gene_list.txt")]
    else:
        stem = Path(gene_list_path).stem
    return os.path.join(output_dir, f"{stem}.pt")


def save_vectors_pt(vectors: dict[str, list[float]], output_path: str) -> None:
    import torch

    tensor_dict = {
        sample_id: torch.tensor(values, dtype=torch.float32)
        for sample_id, values in vectors.items()
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(tensor_dict, output_path)


def read_table(path: str, sep: str | None = None) -> pd.DataFrame:
    if sep is None:
        sep = "\t" if path.lower().endswith((".tsv", ".tsv.gz")) else ","
    return pd.read_csv(path, sep=sep)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a cohort mut_* wide table into the standard OncoWM mutation .pt dictionary."
    )
    parser.add_argument("--input", required=True, help="Path to the cohort molecular CSV/TSV")
    parser.add_argument("--gene-list", default=None, help="Cancer-specific gene list file. Overrides --cancer-type/--list-dir")
    parser.add_argument("--cancer-type", default=None, help="Cancer type used to find list/TCGA-<cancer-type>_gene_list.txt, e.g. KIRP")
    parser.add_argument("--list-dir", default=DEFAULT_LIST_DIR, help=f"Directory for fixed gene lists. Default: {DEFAULT_LIST_DIR}")
    parser.add_argument("--output", default=None, help="Output .pt path. Default: <gene-list-prefix>.pt in --outdir")
    parser.add_argument("--outdir", default="data/processed", help="Used only when --output is not set")
    parser.add_argument("--sample-col", default="sample", help="Sample ID column. Default: sample")
    parser.add_argument("--sep", default=None, help="Input separator. Default: infer from file extension")
    return parser.parse_args()


def main():
    args = parse_args()
    gene_list_path = resolve_gene_list_path(args.gene_list, args.cancer_type, args.list_dir)
    gene_list = load_gene_list(gene_list_path)
    df = read_table(args.input, args.sep)
    vectors = build_mutation_vectors(df, gene_list, sample_col=args.sample_col)
    output_path = args.output or default_output_path(gene_list_path, args.outdir)
    save_vectors_pt(vectors, output_path)

    mut_columns = [col for col in df.columns if col.startswith("mut_")]
    matched_genes = [col.removeprefix("mut_") for col in mut_columns if col.removeprefix("mut_") in set(gene_list)]
    print(f"Samples: {len(vectors)}")
    print(f"Gene list: {gene_list_path}")
    print(f"Gene list length: {len(gene_list)}")
    print(f"mut_ columns in input: {len(mut_columns)}")
    print(f"mut_ columns matched to gene list: {len(matched_genes)}")
    print(f"Saved mutation tensor dictionary to: {output_path}")


if __name__ == "__main__":
    main()
