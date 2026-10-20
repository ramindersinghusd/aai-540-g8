"""Offline validation for Python modules, notebooks, and core ML contracts."""

from __future__ import annotations

import ast
import json
import py_compile
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED_NOTEBOOKS = [
    "AAI540_SageMaker_DataLake_FeatureStore.ipynb",
    "AAI540_SageMaker_Model_Development_Deployment.ipynb",
    "AAI540_SageMaker_ML_System_Observability.ipynb",
    "AAI540_SageMaker_Continual_Learning_CICD.ipynb",
    "AAI540_SageMaker_Retrieval_RAG_Inference.ipynb",
    "AAI540_SageMaker_Application_Integration_Testing.ipynb",
]


def validate_notebooks() -> None:
    for name in EXPECTED_NOTEBOOKS:
        path = ROOT / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            raise AssertionError(f"{name} is not nbformat 4")
        ids = [cell.get("id") for cell in notebook["cells"]]
        if None in ids or len(ids) != len(set(ids)):
            raise AssertionError(f"{name} has missing or duplicate cell IDs")
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell.get("source", []))
            if source.lstrip().startswith(("%", "!")):
                continue
            ast.parse(source, filename=f"{name}:cell-{index}")


def validate_receipt_chain() -> None:
    """Ensure the ordered notebooks retain their explicit S3 handoff contracts."""
    required_markers = {
        "AAI540_SageMaker_DataLake_FeatureStore.ipynb": [
            "run_summary.json",
            '"split_data"',
        ],
        "AAI540_SageMaker_Model_Development_Deployment.ipynb": [
            "run_summary.json",
            "modeling_receipt.json",
        ],
        "AAI540_SageMaker_ML_System_Observability.ipynb": [
            "modeling_receipt.json",
            "monitoring/latest.json",
        ],
        "AAI540_SageMaker_Continual_Learning_CICD.ipynb": [
            "monitoring/latest.json",
            "continual-learning/latest.json",
        ],
        "AAI540_SageMaker_Retrieval_RAG_Inference.ipynb": [
            "monitoring/latest.json",
            "continual-learning/latest.json",
            "retrieval/latest.json",
        ],
        "AAI540_SageMaker_Application_Integration_Testing.ipynb": [
            "retrieval/latest.json",
            "integration/latest.json",
            "cleanup_status",
        ],
    }
    for name, markers in required_markers.items():
        notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
        content = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        missing = [marker for marker in markers if marker not in content]
        if missing:
            raise AssertionError(f"{name} is missing receipt-chain markers: {missing}")


def validate_python() -> None:
    for path in sorted((ROOT / "src").glob("*.py")):
        py_compile.compile(str(path), doraise=True)
    for path in sorted((ROOT / "scripts").glob("*.py")):
        py_compile.compile(str(path), doraise=True)
    for package in ["app", "inference_code"]:
        for path in sorted((ROOT / package).glob("*.py")):
            py_compile.compile(str(path), doraise=True)


def run_smoke_tests() -> None:
    from tests.test_continual_learning import (
        test_candidate_evaluation_gate_accepts_equivalent_model,
        test_prepare_channels_preserves_group_isolation,
    )
    from tests.test_monitor import (
        test_feature_drift_triggers_retraining,
        test_stable_batch_does_not_trigger_retraining,
    )
    from tests.test_train import test_training_artifact_and_batch_inference
    from tests.test_retrieval import (
        test_retrieval_returns_grounded_evidence_and_abstains_unknown_product,
        test_json_inference_contract,
        test_scalar_feature_contract,
    )
    from tests.test_flask_app import test_flask_contract

    test_stable_batch_does_not_trigger_retraining()
    test_feature_drift_triggers_retraining()
    test_prepare_channels_preserves_group_isolation()
    test_scalar_feature_contract()
    test_json_inference_contract()
    test_retrieval_returns_grounded_evidence_and_abstains_unknown_product()
    test_flask_contract()
    with tempfile.TemporaryDirectory() as directory:
        test_candidate_evaluation_gate_accepts_equivalent_model(Path(directory))
    with tempfile.TemporaryDirectory() as directory:
        test_training_artifact_and_batch_inference(Path(directory))


def main() -> None:
    validate_python()
    validate_notebooks()
    validate_receipt_chain()
    run_smoke_tests()
    print(
        "Project validation passed: 6 notebooks, ordered receipt chain, "
        "Python syntax, and 9 core smoke tests."
    )


if __name__ == "__main__":
    main()
