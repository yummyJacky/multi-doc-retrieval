from __future__ import annotations

import csv
import re
from pathlib import Path


def analyze_esg_pages() -> Path:
    """Scan ESG Markdown pages and summarize which pages contain tables and images.

    - Looks for files under `results/dots_ocr/2024_Tencent_ESG` whose names end with digits + `.md`.
    - Marks `has_table` if `<table` appears in the content.
    - Marks `has_image` if a Markdown image like `![...](...png)` appears.
    - Writes a CSV summary file next to the pages directory.
    """

    base_dir = Path(__file__).resolve().parent
    pages_dir = base_dir / "results" / "dots_ocr" / "2024_Tencent_ESG"

    if not pages_dir.is_dir():
        raise FileNotFoundError(f"Pages directory does not exist: {pages_dir}")

    md_pattern = re.compile(r".*(\d+)\.md$")
    image_pattern = re.compile(r"!\[[^\]]*\]\([^)]*\.(?:png|jpg|jpeg|gif|bmp|webp)\)", re.IGNORECASE)

    rows = []

    for md_file in sorted(pages_dir.glob("*.md")):
        # Only keep files whose names end with digits + .md
        if not md_pattern.match(md_file.name):
            continue

        text = md_file.read_text(encoding="utf-8", errors="ignore")

        has_table = "<table" in text and "</table>" in text
        has_image = bool(image_pattern.search(text))

        # Extract page number if pattern like *_page_107.md exists
        m_page = re.search(r"page_(\d+)\.md$", md_file.name)
        page_num = m_page.group(1) if m_page else ""

        rows.append(
            {
                "filename": md_file.name,
                "page_number": page_num,
                "has_table": "1" if has_table else "0",
                "has_image": "1" if has_image else "0",
            }
        )

    # Write summary CSV in the same directory as the pages
    output_path = pages_dir / "2024_Tencent_ESG_page_table_image_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "page_number", "has_table", "has_image"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


if __name__ == "__main__":  # pragma: no cover
    summary_path = analyze_esg_pages()
    print(f"Summary written to: {summary_path}")
