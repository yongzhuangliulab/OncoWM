import pandas as pd

from data_processing.process_clinical_data_qu import (
    build_clinical_prompt,
    normalize_cancer_type,
)
from data_processing.process_mutation_qu import build_mutation_vectors
from data_processing.process_survival_data import build_survival_dataset


def test_qu_cancer_type_mapping():
    assert normalize_cancer_type("肺癌") == "LUAD"
    assert normalize_cancer_type("腺癌") == "LUAD"
    assert normalize_cancer_type("肺腺癌") == "LUAD"
    assert normalize_cancer_type("肺鳞癌") == "LUSC"
    assert normalize_cancer_type("其他") == "unknow"


def test_survival_result_drops_non_numeric_time():
    clinical = pd.DataFrame(
        {
            "样本编号": ["P001", "P002"],
            "随访时间(月)": [12.5, "not available"],
            "随访生存状态\n（存活/死亡）": ["存活", "死亡"],
        }
    )
    pathology = pd.DataFrame(
        {"样本编号": ["P001", "P002"], "分组": ["肺癌", "鳞癌"]}
    )

    result = build_survival_dataset(clinical, pathology)

    assert result.to_dict("records") == [
        {
            "sample_id": "P001",
            "cancer_type": "LUAD",
            "survival_time": 12.5,
            "survival_event": 0,
        }
    ]


def test_survival_result_drops_unrecognized_event_by_default():
    clinical = pd.DataFrame(
        {
            "样本编号": ["P001"],
            "随访时间(月)": [12.5],
            "随访生存状态\n（存活/死亡）": ["未记录"],
        }
    )
    pathology = pd.DataFrame({"样本编号": ["P001"], "分组": ["肺癌"]})

    assert build_survival_dataset(clinical, pathology).empty


def test_clinical_prompt_keeps_original_fields_and_excludes_id():
    row = pd.Series(
        {
            "样本编号": "SECRET-ID",
            "分组": "肺癌",
            "性别": "女",
            "年龄": 62,
            "TNM 分期": "IIA",
        }
    )

    prompt = build_clinical_prompt(row)

    assert "Cancer Subtype: LUAD" in prompt
    assert "Age: 62" in prompt
    assert "Sex: 女" in prompt
    assert "AJCC Pathologic Stage: IIA" in prompt
    assert "SECRET-ID" not in prompt


def test_mutation_vectors_use_only_mut_columns_and_gene_list_order():
    frame = pd.DataFrame(
        {
            "sample": ["P001"],
            "mut_EGFR": [1],
            "mut_TP53": [0],
            "cnv_TP53": [1],
        }
    )

    vectors = build_mutation_vectors(frame, ["TP53", "EGFR", "KRAS"])

    assert vectors == {"P001": [0.0, 1.0, 0.0]}
