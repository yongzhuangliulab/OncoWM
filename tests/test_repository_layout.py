import ast
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_model_api_has_no_unused_moe_parameters():
    source = (ROOT / "src/models/OncoWM.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    onco_jepa = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "OncoJEPA"
    )
    constructor = next(
        node for node in onco_jepa.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameter_names = {argument.arg for argument in constructor.args.args}

    assert "num_experts" not in parameter_names
    assert "k_expert" not in parameter_names
    assert "self.out_proj" not in source


def test_documentation_uses_oncowm_name_and_valid_local_links():
    legacy_names = re.compile(r"\bM?-?JEPA\b|Multimoda_JEPA", re.IGNORECASE)
    markdown_files = sorted(ROOT.rglob("*.md"))

    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        assert not legacy_names.search(text), path

        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("http://", "https://", "#")):
                continue
            local_target = target.split("#", 1)[0]
            if local_target:
                assert (path.parent / local_target).is_file(), (path, target)


def test_example_configs_contain_required_sections():
    train = yaml.safe_load((ROOT / "configs/train.example.yaml").read_text(encoding="utf-8"))
    inference = yaml.safe_load((ROOT / "configs/inference.example.yaml").read_text(encoding="utf-8"))

    assert {"data", "models", "optimization", "meta"}.issubset(train)
    assert train["models"]["text_encoder_name"] == "pretrained_models/medgemma-1.5-4b-it"
    assert {"task_mode", "model_path", "output_path", "data", "models"}.issubset(inference)
    assert inference["model_path"].startswith("pretrained_models/oncowm/")
    assert inference["models"]["text_encoder_name"] == "pretrained_models/medgemma-1.5-4b-it"
    assert "MoE" not in train
    assert "MoE" not in inference
    assert {"hidden_size", "dropout_rate"}.issubset(train["models"])
    assert "hidden_size" in inference["models"]
    assert "dropout_rate" not in inference["models"]
    assert "folder" not in train["logging"]
    assert set(train["logging"]) == {"write_tag"}


def test_all_dfi_inference_example_uses_portable_layout():
    path = ROOT / "configs/inference.ALL_DFI_full_cohort.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["run_name"] == "full_cohort"
    assert config["model_path"] == (
        "pretrained_models/oncowm/ALL_DFI/full_cohort/prognosis_full_final.pth.tar"
    )
    assert config["data"]["csv_path"] == (
        "data/processed/tcga/survival/ALL_DFI_dataset.csv"
    )
    assert "MoE" not in config
    assert config["models"]["hidden_size"] == 1024
    assert "dropout_rate" not in config["models"]
    assert config["models"]["geno_raw_dim"] == 19766
    assert (ROOT / "USAGE.md").is_file()


def test_tcga_configs_are_complete_and_portable():
    configs = sorted((ROOT / "configs/tcga").glob("train_*.yaml"))
    assert len(configs) == 84

    required_sections = {"data", "models", "optimization", "meta", "logging", "tb"}
    expected_top_level = required_sections | {"output_path", "seed", "num_folds"}
    expected_endpoints = {"OS", "DSS", "DFI", "PFI"}
    endpoints_by_cancer = {}

    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(config) == expected_top_level, path
        _, cancer, endpoint = path.stem.split("_")
        endpoints_by_cancer.setdefault(cancer, set()).add(endpoint)

        assert config["num_folds"] == 5
        assert isinstance(config["data"]["align_batch_size"], int)
        assert isinstance(config["data"]["prognosis_batch_size"], int)
        assert isinstance(config["models"]["geno_raw_dim"], int)
        assert config["models"]["hidden_size"] == 1024
        assert config["models"]["dropout_rate"] == 0.1
        assert "MoE" not in config
        assert isinstance(config["optimization"]["t2i_weight_decay"], (int, float))
        assert isinstance(config["optimization"]["prognosis"]["lr"], (int, float))
        assert config["data"]["csv_path"].startswith("data/processed/tcga/")
        assert config["output_path"].startswith("outputs/train/")

    assert len(endpoints_by_cancer) == 21
    assert all(endpoints == expected_endpoints for endpoints in endpoints_by_cancer.values())


def test_publishable_text_has_no_original_server_paths():
    forbidden = ("/home/Disk", "C:\\Users\\62519", "cuda:1")
    suffixes = {".py", ".md", ".yaml", ".yml", ".sh"}
    package_paths = [
        ROOT / "README.md",
        ROOT / "README_EN.md",
        ROOT / "USAGE.md",
        ROOT / "USAGE_EN.md",
        ROOT / "main.py",
        ROOT / "environment.yml",
    ]
    for directory in (
        "configs",
        "data",
        "data_processing",
        "docs",
        "pretrained_models",
        "scripts",
        "src",
    ):
        package_paths.extend((ROOT / directory).rglob("*"))

    for path in package_paths:
        if not path.is_file() or path.suffix.lower() not in suffixes or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path
