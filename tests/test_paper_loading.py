"""Tests for paper-as-oracle loading.

Verifies that paper.md is read into PaperConfig.paper_text and that the
absence of paper.md is handled gracefully (warning, not crash).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.core.config_loader import load_paper_config


@pytest.fixture
def fake_paper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a fake config/papers/<id>/ tree under tmp_path and point
    config_loader at it via get_project_root."""
    paper_id = "fake_paper"
    paper_dir = tmp_path / "config" / "papers" / paper_id
    paper_dir.mkdir(parents=True)

    config = {
        "paper": {
            "id": paper_id,
            "title": "A Fake Paper",
            "authors": "Nobody et al.",
            "url": "https://example.com",
            "original_model": "TestModel-7B",
            "model_variant": "instruct",
            "paper_text_path": "paper.md",
        },
        "techniques_required": ["probes"],
        "claims": [
            {
                "id": "claim_1",
                "description": "Probes classify 2 concepts above chance",
                "experiment_type": "probe_classification",
                "params": {"concept_set": ["a", "b"]},
                "success_metric": "probe_accuracy",
                "success_threshold": 0.7,
                "paper_section": "Section 3.1",
            }
        ],
    }
    (paper_dir / "paper_config.yaml").write_text(yaml.safe_dump(config))

    # Patch get_project_root so config_loader looks under tmp_path.
    import src.core.config_loader as cl

    monkeypatch.setattr(cl, "get_project_root", lambda: tmp_path)
    return paper_dir


def test_paper_text_loaded_when_present(fake_paper: Path) -> None:
    paper_md = "# Fake Paper\n\nThis is the full text of the paper.\n"
    (fake_paper / "paper.md").write_text(paper_md)

    config = load_paper_config("fake_paper")

    assert config.paper_text == paper_md
    assert config.paper_text_path == "paper.md"
    assert config.title == "A Fake Paper"


def test_paper_text_empty_when_missing(
    fake_paper: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No paper.md should produce an empty paper_text and a warning, not
    a crash."""
    config = load_paper_config("fake_paper")

    assert config.paper_text == ""
    assert any(
        "No paper text found" in record.message for record in caplog.records
    ), "Expected a warning about missing paper text"


def test_paper_section_propagates_to_claims(fake_paper: Path) -> None:
    (fake_paper / "paper.md").write_text("# Fake Paper\n")
    config = load_paper_config("fake_paper")

    assert len(config.claims) == 1
    assert config.claims[0].paper_section == "Section 3.1"


def test_custom_paper_text_path(fake_paper: Path) -> None:
    """If paper_text_path is set to a non-default name, that file is read."""
    raw = yaml.safe_load((fake_paper / "paper_config.yaml").read_text())
    raw["paper"]["paper_text_path"] = "subdir/the_paper.md"
    (fake_paper / "paper_config.yaml").write_text(yaml.safe_dump(raw))

    (fake_paper / "subdir").mkdir()
    (fake_paper / "subdir" / "the_paper.md").write_text("# Custom path\n")

    config = load_paper_config("fake_paper")
    assert "Custom path" in config.paper_text
