from __future__ import annotations

import re
from pathlib import Path

import nbformat
import matplotlib.pyplot as plt

from scripts.build_notebook import build_notebook
from scripts.build_report import build_report


ROOT = Path(__file__).resolve().parents[1]


def test_generated_notebook_has_required_sections_and_compilable_code(tmp_path: Path) -> None:
    output = tmp_path / "laboratorio.ipynb"

    build_notebook(output)
    notebook = nbformat.read(output, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")

    for heading in (
        "# Laboratorio de series de tiempo",
        "## 1. Datos",
        "## 2. Exploración y preparación",
        "## 3. Investigación",
        "## 4. Protocolo de validación y métricas",
        "## 5. Construcción y entrenamiento",
        "### 5.1 Estrategias multi-paso",
        "### 5.2 Iteraciones de hiperparámetros",
        "### 5.3 Modelo local contra global",
        "## 6. Comparación de resultados",
        "## 7. Artefactos reproducibles",
        "## 8. Discusión y análisis",
    ):
        assert heading in markdown
    assert "Persona A" not in markdown
    assert "Persona B" not in markdown
    namespace: dict[str, object] = {"__name__": "__notebook_smoke__"}
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(output), "exec")
            exec(cell.source, namespace)


def test_report_is_at_most_five_pages_and_contains_repository_link(tmp_path: Path) -> None:
    source = ROOT / "report" / "reporte.md"
    output = tmp_path / "reporte.pdf"

    build_report(source, output)
    payload = output.read_bytes()
    page_count = len(re.findall(rb"/Type\s*/Page\b", payload))

    assert payload.startswith(b"%PDF")
    assert 1 <= page_count <= 5
    assert b"github.com/JosFer720/Laboratorio-9-Series-de-tiempo" in payload


def test_report_embeds_figure_markers(tmp_path: Path) -> None:
    """Las gráficas declaradas en Markdown deben aparecer dentro del PDF."""
    figure_path = tmp_path / "figura.png"
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    figure.savefig(figure_path)
    plt.close(figure)
    source = tmp_path / "reporte.md"
    source.write_text(
        "# Reporte\n\n<!-- FIGURE: figura.png -->\n", encoding="utf-8"
    )

    output = build_report(source, tmp_path / "reporte.pdf")

    assert b"/Subtype /Image" in output.read_bytes()


def test_repository_text_does_not_expose_work_assignment() -> None:
    checked = [ROOT / "README.md", ROOT / "report" / "reporte.md"]
    for path in checked:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            assert "Persona A" not in text
            assert "Persona B" not in text


def test_delivered_notebook_matches_the_generator(tmp_path: Path) -> None:
    generated = tmp_path / "laboratorio.ipynb"
    build_notebook(generated)

    assert generated.read_text(encoding="utf-8") == (ROOT / "laboratorio_series_tiempo.ipynb").read_text(
        encoding="utf-8"
    )
