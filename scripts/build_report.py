"""Renderiza el reporte Markdown como un PDF de hasta cinco páginas."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPOSITORY_URL = "https://github.com/JosFer720/Laboratorio-9-Series-de-tiempo"


def _plain_lines(markdown: str) -> list[tuple[str, str]]:
    """Convierte Markdown básico en líneas tipadas aptas para Matplotlib."""

    lines: list[tuple[str, str]] = []
    for raw in markdown.splitlines():
        value = raw.strip()
        if not value:
            lines.append(("espacio", ""))
            continue
        if value.startswith("# "):
            lines.append(("titulo", value[2:]))
        elif value.startswith("## "):
            lines.append(("subtitulo", value[3:]))
        else:
            cleaned = re.sub(r"[*`]", "", value)
            width = 102 if cleaned.startswith("Repositorio:") else 94
            for part in textwrap.wrap(cleaned, width=width, break_long_words=False):
                lines.append(("texto", part))
    return lines


def build_report(source: str | Path, output: str | Path) -> Path:
    """Construye un PDF y rechaza una fuente con más de cinco páginas."""

    source_path = Path(source)
    pages = source_path.read_text(encoding="utf-8").split("<!-- PAGE BREAK -->")
    if len(pages) > 5:
        raise ValueError("El reporte no puede exceder cinco páginas")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous_compression = mpl.rcParams["pdf.compression"]
    mpl.rcParams["pdf.compression"] = 0
    try:
        metadata = {"Title": "Laboratorio de series de tiempo", "Subject": REPOSITORY_URL}
        with PdfPages(destination, metadata=metadata) as pdf:
            for number, page in enumerate(pages, start=1):
                figure_paths = re.findall(r"<!--\s*FIGURE:\s*(.+?)\s*-->", page)
                page = re.sub(r"<!--\s*FIGURE:\s*.+?\s*-->", "", page)
                figure = plt.figure(figsize=(8.27, 11.69))
                axis = figure.add_axes((0.08, 0.06, 0.84, 0.88))
                axis.axis("off")
                y = 0.98
                for kind, line in _plain_lines(page):
                    if kind == "espacio":
                        y -= 0.018
                        continue
                    size = 17 if kind == "titulo" else 12 if kind == "subtitulo" else 9.2
                    weight = "bold" if kind in {"titulo", "subtitulo"} else "normal"
                    axis.text(0, y, line, fontsize=size, fontweight=weight, va="top", family="DejaVu Sans")
                    y -= 0.047 if kind == "titulo" else 0.035 if kind == "subtitulo" else 0.024
                for index, relative_path in enumerate(figure_paths):
                    image_path = (source_path.parent / relative_path.strip()).resolve()
                    if not image_path.exists():
                        raise FileNotFoundError(f"No existe la figura declarada: {image_path}")
                    count = len(figure_paths)
                    width = 0.82 if count == 1 else 0.39
                    left = 0.09 if count == 1 else 0.09 + index * 0.42
                    image_axis = figure.add_axes((left, 0.08, width, 0.34))
                    image_axis.imshow(plt.imread(image_path))
                    image_axis.axis("off")
                axis.text(1, 0, f"{number}/{len(pages)}", ha="right", va="bottom", fontsize=8, color="#555555")
                pdf.savefig(figure)
                plt.close(figure)
    finally:
        mpl.rcParams["pdf.compression"] = previous_compression
    return destination


if __name__ == "__main__":
    build_report("report/reporte.md", "report/reporte_laboratorio.pdf")
