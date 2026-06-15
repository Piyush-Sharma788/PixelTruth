import re
from pathlib import Path

import pytest


APP_PY = Path(__file__).resolve().parents[1] / "app.py"


def _find_st_image_calls(text: str) -> list[str]:
    pattern = r'st\.image\(\s*"([^"]+)"'
    return re.findall(pattern, text)


def test_no_duplicate_st_image_calls_for_training_plots():
    if not APP_PY.exists():
        pytest.skip("app.py not found")

    source = APP_PY.read_text(encoding="utf-8")
    figure_files = [path for path in _find_st_image_calls(source) if "Figure" in path]

    seen = {}
    for path in figure_files:
        if path in seen:
            raise AssertionError(
                f"Duplicate st.image call for '{path}' found "
                f"(first occurrence at image reference #{seen[path]}, "
                f"second at #{figure_files.index(path) + 1})"
            )
        seen[path] = figure_files.index(path) + 1


def test_training_plots_use_use_container_width():
    if not APP_PY.exists():
        pytest.skip("app.py not found")

    source = APP_PY.read_text(encoding="utf-8")
    figure_lines = [
        line.strip()
        for line in source.splitlines()
        if "st.image" in line and "Figure" in line
    ]

    for line in figure_lines:
        assert "use_container_width=True" in line, (
            f"Line uses deprecated use_column_width instead of use_container_width:\n{line}"
        )
        assert "use_column_width" not in line, (
            f"Line still contains deprecated use_column_width:\n{line}"
        )


def test_training_plots_have_no_deprecated_use_column_width():
    if not APP_PY.exists():
        pytest.skip("app.py not found")

    source = APP_PY.read_text(encoding="utf-8")
    for i, line in enumerate(source.splitlines(), 1):
        if "st.image" in line and "Figure" in line:
            assert "use_column_width" not in line, (
                f"Line {i} uses deprecated use_column_width:\n{line}"
            )
