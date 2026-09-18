#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RELEASE_ROOT="${ROOT_DIR}/release"
ARXIV_DIR="${RELEASE_ROOT}/arxiv"
ZIP_PATH="${RELEASE_ROOT}/structviz-bench-arxiv.zip"

MAIN_TEX_SRC="${ROOT_DIR}/paper/main.tex"
REFERENCES_SRC="${ROOT_DIR}/paper/references.bib"
STYLE_SRC="${ROOT_DIR}/paper/neurips_2025.sty"
QUALITATIVE_SRC="${ROOT_DIR}/results/qualitative_examples.tex"
FIGURES_DIR="${ROOT_DIR}/paper/figures_full"

for required_file in "${MAIN_TEX_SRC}" "${REFERENCES_SRC}" "${STYLE_SRC}" "${QUALITATIVE_SRC}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

if [[ ! -d "${FIGURES_DIR}" ]]; then
  echo "Missing figure directory: ${FIGURES_DIR}" >&2
  exit 1
fi

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex not found in PATH" >&2
  exit 1
fi

if ! command -v bibtex >/dev/null 2>&1; then
  echo "bibtex not found in PATH" >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "zip not found in PATH" >&2
  exit 1
fi

mkdir -p "${RELEASE_ROOT}"
rm -rf "${ARXIV_DIR}"
rm -f "${ZIP_PATH}"
mkdir -p "${ARXIV_DIR}"

cp "${MAIN_TEX_SRC}" "${ARXIV_DIR}/main.tex"
cp "${REFERENCES_SRC}" "${ARXIV_DIR}/references.bib"
cp "${STYLE_SRC}" "${ARXIV_DIR}/neurips_2025.sty"
cp "${QUALITATIVE_SRC}" "${ARXIV_DIR}/qualitative_examples.tex"

figure_count=0
for figure in "${FIGURES_DIR}"/*.pdf; do
  if [[ -f "${figure}" ]]; then
    cp "${figure}" "${ARXIV_DIR}/$(basename "${figure}")"
    figure_count=$((figure_count + 1))
  fi
done

if [[ "${figure_count}" -eq 0 ]]; then
  echo "No PDF figures found in ${FIGURES_DIR}" >&2
  exit 1
fi

export ARXIV_MAIN_TEX="${ARXIV_DIR}/main.tex"
python3 - <<'PY'
from pathlib import Path
import os

main_tex = Path(os.environ["ARXIV_MAIN_TEX"])
text = main_tex.read_text(encoding="utf-8")
text = text.replace("figures_full/", "")
text = text.replace("../results/qualitative_examples", "qualitative_examples")
main_tex.write_text(text, encoding="utf-8")
PY

(
  cd "${ARXIV_DIR}"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  bibtex main >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
)

rm -f "${ARXIV_DIR}/main.aux" \
      "${ARXIV_DIR}/main.bbl" \
      "${ARXIV_DIR}/main.blg" \
      "${ARXIV_DIR}/main.log" \
      "${ARXIV_DIR}/main.out" \
      "${ARXIV_DIR}/main.pdf"

(
  cd "${ARXIV_DIR}"
  zip -r "${ZIP_PATH}" . >/dev/null
)

echo "arXiv package created: ${ZIP_PATH}"
echo "Copied PDF figures: ${figure_count}"
echo "Included files:"
(
  cd "${ARXIV_DIR}"
  ls -1
)
zip_size_bytes=$(stat -c%s "${ZIP_PATH}")
echo "Zip size (bytes): ${zip_size_bytes}"
