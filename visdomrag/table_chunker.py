import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


QUESTION_TYPE_FACT = "fact"
QUESTION_TYPE_ANALYSIS = "analysis"


@dataclass
class TableFactChunk:
    """Structured chunk extracted from a table for factual QA.

    Each chunk contains a short natural-language text plus a metadata dict
    whose schema depends on the specific table type.
    """

    text: str
    metadata: Dict[str, Any]


def classify_question_type(question: str) -> str:
    """Classify a user question into factual vs analysis.

    This uses a lightweight heuristic based on Chinese patterns and the
    presence of numbers / percentage markers. It is intentionally simple
    and deterministic so it can run before any LLM calls.
    """

    if not isinstance(question, str):
        return QUESTION_TYPE_ANALYSIS

    q = question.strip()
    if not q:
        return QUESTION_TYPE_ANALYSIS

    # Markers that very often correspond to numeric / factual questions
    fact_markers = [
        "多少",
        "几",  # e.g. 几个、几年
        "多大",
        "数量",
        "人数",
        "比率",
        "比例",
        "百分比",
        "%",
    ]
    for m in fact_markers:
        if m in q:
            return QUESTION_TYPE_FACT

    # If question contains explicit digits and does not obviously ask for
    # explanation / reasoning, treat as factual.
    if re.search(r"\d", q):
        analysis_markers = [
            "为什么",
            "如何",
            "原因",
            "影响",
            "分析",
            "比较",
            "对比",
            "趋势",
            "评价",
            "说明",
        ]
        if not any(m in q for m in analysis_markers):
            return QUESTION_TYPE_FACT

    return QUESTION_TYPE_ANALYSIS


def _extract_tables_from_markdown(md_text: str) -> List[str]:
    """Extract raw HTML <table>...</table> blocks from a markdown page.

    The markdown produced by OCR contains tables rendered as HTML blocks.
    """

    if not md_text:
        return []
    pattern = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
    return pattern.findall(md_text)


def _strip_html_tags(text: str) -> str:
    """Remove basic HTML tags from a small snippet of text."""

    if not text:
        return ""
    # Remove tags like <sup>...</sup>, <br>, etc.
    return re.sub(r"<[^>]+>", "", text).strip()


def _normalize_year_cell(cell: str) -> Optional[str]:
    """Convert a header cell like "二零二四年" into "2024" if possible."""

    if not cell:
        return None
    cell = _strip_html_tags(cell)
    cell = cell.replace("年", "").strip()
    if not cell:
        return None

    # Map common Chinese numerals to digits
    digit_map = {
        "零": "0",
        "〇": "0",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
    }

    # If it already contains digits, keep only digits
    if re.search(r"\d", cell):
        digits = re.findall(r"\d", cell)
        return "".join(digits) if digits else None

    # Otherwise map Chinese numerals
    digits: List[str] = []
    for ch in cell:
        if ch in digit_map:
            digits.append(digit_map[ch])
    if not digits:
        return None
    return "".join(digits)


def _is_numeric_cell(text: str) -> bool:
    """Heuristic: treat as numeric if it contains digits or a percent sign."""

    if not text:
        return False
    t = _strip_html_tags(text)
    return bool(re.search(r"[0-9]", t) or "%" in t)


def _detect_employee_table(table_html: str) -> bool:
    """Detect whether a table looks like the employee scale/turnover table.

    We key off a few characteristic phrases to avoid mis-parsing unrelated
    tables.
    """

    if not table_html:
        return False
    text = _strip_html_tags(table_html)
    keywords = [
        "员工总数",
        "员工人数",
        "流失率",
        "按性别划分的员工人数",
        "按年龄组别划分的员工人数",
        "按地理区域划分的员工人数",
    ]
    return any(k in text for k in keywords)


def _chunk_employee_table(
    table_html: str,
    doc_id: str,
    page_number: int,
    source: Optional[str] = None,
) -> List[TableFactChunk]:
    """Chunk an "员工规模与流失率"风格的表格为原子事实.

    This implements the state-machine logic discussed in the design:
    - Step 1: parse header to get years.
    - Step 2: iterate tbody rows, maintaining current_metric/current_dimension.
    - Step 3: for each data row, create one chunk per (year, value).

    The metadata schema here is specific to this table family only.
    """

    chunks: List[TableFactChunk] = []
    if not table_html:
        return chunks

    # Extract thead and tbody blocks
    thead_match = re.search(r"<thead>(.*?)</thead>", table_html, re.IGNORECASE | re.DOTALL)
    tbody_match = re.search(r"<tbody>(.*?)</tbody>", table_html, re.IGNORECASE | re.DOTALL)
    if not thead_match or not tbody_match:
        return chunks

    thead_html = thead_match.group(1)
    tbody_html = tbody_match.group(1)

    # Header: we expect two rows, second row contains year columns
    header_rows = re.findall(r"<tr>(.*?)</tr>", thead_html, re.IGNORECASE | re.DOTALL)
    years: List[str] = []
    if len(header_rows) >= 2:
        second_row = header_rows[1]
        year_cells = re.findall(r"<th[^>]*>(.*?)</th>", second_row, re.IGNORECASE | re.DOTALL)
        for cell in year_cells:
            year = _normalize_year_cell(cell)
            if year:
                years.append(year)

    if not years:
        # Fallback: try all header th cells
        header_cells = re.findall(r"<th[^>]*>(.*?)</th>", thead_html, re.IGNORECASE | re.DOTALL)
        for cell in header_cells:
            year = _normalize_year_cell(cell)
            if year:
                years.append(year)

    if not years:
        return chunks

    # State machine over tbody rows
    current_metric: Optional[str] = None  # e.g. "员工人数" / "员工流失率"
    current_dimension: Optional[str] = None  # e.g. "性别" / "年龄组别"
    current_group: Optional[str] = None  # e.g. "按性别划分的员工人数"

    row_html_list = re.findall(r"<tr>(.*?)</tr>", tbody_html, re.IGNORECASE | re.DOTALL)
    for row_html in row_html_list:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.IGNORECASE | re.DOTALL)
        if not cells:
            continue

        cell_texts = [_strip_html_tags(c) for c in cells]
        first_col = cell_texts[0].strip() if cell_texts else ""
        other_cols = cell_texts[1:]

        has_numeric_other = any(_is_numeric_cell(c) for c in other_cols)

        # Group/title row: first column has text, others are not numeric
        if first_col and not has_numeric_other:
            title = first_col
            current_group = title

            # Metric
            if "流失率" in title:
                current_metric = "员工流失率"
            elif "员工人数" in title or "员工总数" in title:
                current_metric = "员工人数"

            # Dimension
            if "按年龄组别" in title:
                current_dimension = "年龄组别"
            elif "按性别" in title:
                current_dimension = "性别"
            elif "按地理区域" in title:
                current_dimension = "地理区域"
            elif "按雇佣类型" in title:
                current_dimension = "雇佣类型"

            continue

        # Data row: first column text + numeric values in other columns
        if not first_col or not has_numeric_other:
            continue

        dimension_value = first_col
        values = other_cols

        # Decide metric based on current state and value patterns
        value_has_percent = any("%" in v for v in values)
        metric = current_metric
        if metric is None:
            metric = "员工流失率" if value_has_percent else "员工人数"

        dimension = current_dimension or "整体"

        for idx, year in enumerate(years):
            if idx >= len(values):
                continue
            raw_val = values[idx].strip()
            if not _is_numeric_cell(raw_val):
                continue

            unit = "%" if "%" in raw_val else "人"

            # Natural-language text for retrieval
            group_prefix = f"{current_group}中" if current_group else ""
            text = (
                f"{year}年，{group_prefix}{dimension}为{dimension_value}的{metric}为{raw_val}。"
            )

            metadata: Dict[str, Any] = {
                "metric": metric,
                "dimension": dimension,
                "dimension_value": dimension_value,
                "year": int(year) if year.isdigit() else year,
                "unit": unit,
                "source": source or "员工规模与流失率表",
                "group_title": current_group,
                "doc_id": doc_id,
                "page_number": page_number,
            }

            chunks.append(TableFactChunk(text=text, metadata=metadata))

    return chunks


def chunk_tables_for_fact_questions(
    md_text: str,
    doc_id: str,
    page_number: int,
    question_type: str,
    source: Optional[str] = None,
) -> List[TableFactChunk]:
    """Chunk tables on a page into atomic factual statements when appropriate.

    This function is intended to be called after OCR, and only when the
    question has been classified as factual. It detects table types and
    dispatches to table-specific chunkers with their own metadata schemas.
    """

    if question_type != QUESTION_TYPE_FACT:
        return []

    tables = _extract_tables_from_markdown(md_text)
    if not tables:
        return []

    all_chunks: List[TableFactChunk] = []
    for table_html in tables:
        if _detect_employee_table(table_html):
            all_chunks.extend(
                _chunk_employee_table(table_html, doc_id, page_number, source=source)
            )
        # Future: other table types can be added here with their own
        # detection logic and metadata schema.

    return all_chunks
