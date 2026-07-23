from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_example_configs_contain_required_sections():
    train = yaml.safe_load((ROOT / "configs/train.example.yaml").read_text(encoding="utf-8"))
    inference = yaml.safe_load((ROOT / "configs/inference.example.yaml").read_text(encoding="utf-8"))

    assert {"data", "models", "MoE", "optimization", "meta"}.issubset(train)
    assert train["models"]["text_encoder_name"] == "pretrained_models/medgemma-1.5-4b-it"
    assert {"task_mode", "model_path", "output_path", "data", "models", "MoE"}.issubset(inference)
    assert inference["model_path"].startswith("pretrained_models/mjepa/")
    assert inference["models"]["text_encoder_name"] == "pretrained_models/medgemma-1.5-4b-it"


def test_all_dfi_inference_example_uses_portable_layout():
    path = ROOT / "configs/inference.ALL_DFI_fold_0.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["run_name"] == "fold_0"
    assert config["model_path"] == (
        "pretrained_models/mjepa/ALL_DFI/fold_0/prognosis-best.pth.tar"
    )
    assert config["data"]["csv_path"] == (
        "data/processed/tcga/survival/ALL_DFI_fold_0_val.csv"
    )
    assert config["models"]["geno_raw_dim"] == 19766
    assert (ROOT / "USAGE.md").is_file()


def test_tcga_configs_are_complete_and_portable():
    configs = sorted((ROOT / "configs/tcga").glob("train_*.yaml"))
    assert len(configs) == 84

    required_sections = {"data", "models", "MoE", "optimization", "meta", "logging", "tb"}
    expected_endpoints = {"OS", "DSS", "DFI", "PFI"}
    endpoints_by_cancer = {}

    for path in configs:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert required_sections.issubset(config), path
        _, cancer, endpoint = path.stem.split("_")
        endpoints_by_cancer.setdefault(cancer, set()).add(endpoint)

        assert config["num_folds"] == 5
        assert isinstance(config["data"]["align_batch_size"], int)
        assert isinstance(config["data"]["prognosis_batch_size"], int)
        assert isinstance(config["models"]["geno_raw_dim"], int)
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
        ROOT / "USAGE.md",
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
