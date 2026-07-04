"""Render reports/report.md to a styled PDF via Playwright/Chromium (Phase 18).

Markdown is converted to HTML, wrapped in a print stylesheet, written next to the
figures so relative image paths resolve, and printed to PDF by headless Chromium.
Run inside the project virtualenv (requires `playwright install chromium`).
"""

from __future__ import annotations

from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MD_PATH = REPORTS / "report.md"
PDF_PATH = REPORTS / "report.pdf"
TMP_HTML = REPORTS / "_report_render.html"

# Serif body for prose, sans headings; sections start on a fresh page and
# figures/tables never split across a page break.
CSS = """
@page { margin: 18mm 15mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 11pt;
       line-height: 1.5; color: #1a1a1a; }
h1, h2, h3, h4 { font-family: 'Helvetica Neue', Arial, sans-serif; color: #14314f;
                 page-break-after: avoid; }
h1 { font-size: 22pt; }
h2 { font-size: 16pt; border-bottom: 2px solid #2c7fb8; padding-bottom: 4px;
     page-break-before: always; margin-top: 0; }
h3 { font-size: 13pt; }
h2:first-of-type { page-break-before: avoid; }
a { color: #2c7fb8; text-decoration: none; }
img { max-width: 100%; display: block; margin: 12px auto; page-break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 10px 0;
        page-break-inside: avoid; }
th, td { border: 1px solid #cbd5e0; padding: 4px 8px; text-align: left; }
th { background: #f0f4f8; }
code { background: #f0f4f8; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
hr { border: none; border-top: 1px solid #cbd5e0; margin: 16px 0; }
"""

FOOTER = (
    '<div style="font-size:8px;width:100%;text-align:center;color:#888;">'
    '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
)


def build_html() -> str:
    """Convert the report markdown to a self-contained styled HTML document."""
    html_body = markdown.markdown(
        MD_PATH.read_text(encoding="utf-8"),
        extensions=["tables", "toc", "fenced_code", "attr_list"],
    )
    return f"<!doctype html><html><head><meta charset='utf-8'>" \
           f"<style>{CSS}</style></head><body>{html_body}</body></html>"


def main() -> int:
    TMP_HTML.write_text(build_html(), encoding="utf-8")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(TMP_HTML.as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(PDF_PATH),
                format="A4",
                print_background=True,
                margin={"top": "18mm", "bottom": "18mm", "left": "15mm", "right": "15mm"},
                display_header_footer=True,
                header_template="<div></div>",
                footer_template=FOOTER,
            )
            browser.close()
    finally:
        TMP_HTML.unlink(missing_ok=True)

    size_kb = PDF_PATH.stat().st_size / 1024
    print(f"wrote {PDF_PATH.relative_to(ROOT)} ({size_kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
