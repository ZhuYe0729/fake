from pathlib import Path
import subprocess
import sys

import pytest

from extras.common.config import load_config
from extras.fakevlm.dataset import classify_response, score_predictions


def test_extra_config_requires_model_and_dataset(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[model]\npath="model"\n[dataset]\npath="data"\n')
    assert load_config(config)["model"]["path"] == "model"


def test_fakeclue_scoring_preserves_label_convention() -> None:
    assert classify_response("The image is real.") == 1
    assert classify_response("This looks fake.") == 0
    assert classify_response("uncertain") is None
    result = score_predictions([1, 0, 1], [1, 0, None])
    assert result["accuracy"] == 2 / 3
    assert result["unknown_responses"] == 1


@pytest.mark.parametrize("module", [
    "extras.maxvit.evaluate", "extras.dinov3.evaluate",
    "extras.mirror.evaluate", "extras.fakevlm.evaluate",
])
def test_documented_extra_cli_entrypoints(module: str) -> None:
    subprocess.run([sys.executable, "-m", module, "--help"], check=True,
                   stdout=subprocess.DEVNULL)
