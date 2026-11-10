import pathlib
"""Chuyển báo cáo Markdown sang DOCX (đơn giản, đủ dùng để nộp)."""
import re
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / "BAO-CAO-Sutton-Barto.md"
DST = HERE / "BAO-CAO-Sutton-Barto.docx"

doc = Document()

# --- style chung ---
style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
style.paragraph_format.space_after = Pt(6)
style.paragraph_format.line_spacing = 1.25

for section in doc.sections:
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.2)
    section.left_margin = Cm(2.8)
    section.right_margin = Cm(2.2)

INLINE = re.compile(r"(\*\*.+?\*\*|`.+?`|\*[^*]+\*)")


def add_inline(par, text):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`") and part.endswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(10.5)
        elif part.startswith("*") and part.endswith("*"):
            r = par.add_run(part[1:-1])
            r.italic = True
        else:
            par.add_run(part)


def add_code(lines):
    par = doc.add_paragraph()
    par.paragraph_format.left_indent = Cm(0.6)
    par.paragraph_format.space_before = Pt(4)
    par.paragraph_format.space_after = Pt(8)
    par.paragraph_format.line_spacing = 1.0
    run = par.add_run("\n".join(lines))
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


def add_table(rows):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = []
    for line in rows[1:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue
        body.append(cells)

    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, h in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(10.5)

    for row in body:
        cells = table.add_row().cells
        for i, val in enumerate(row[: len(header)]):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            add_inline(p, val)
            for run in p.runs:
                run.font.size = Pt(10)

    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_heading(text, level, size=None, color=None):
    h = doc.add_heading("", level=level)
    add_inline(h, text)
    for r in h.runs:
        r.font.name = "Times New Roman"
        if size:
            r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    return h


lines = open(SRC, encoding="utf-8").read().splitlines()
i = 0
while i < len(lines):
    line = lines[i].rstrip()

    # code block
    if line.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].startswith("```"):
            buf.append(lines[i])
            i += 1
        add_code(buf)
        i += 1
        continue

    # table
    if line.startswith("|"):
        buf = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            buf.append(lines[i])
            i += 1
        add_table(buf)
        continue

    if line.startswith("# "):
        add_heading(line[2:], 0, size=20)
        i += 1
        continue

    if line.startswith("## "):
        add_heading(line[3:], 1, color=RGBColor(0x1F, 0x3B, 0x73))
        i += 1
        continue

    if line.startswith("### "):
        add_heading(line[4:], 2, color=RGBColor(0x1F, 0x3B, 0x73))
        i += 1
        continue

    if line.strip() == "---":
        i += 1
        continue

    if line.startswith("- "):
        p = doc.add_paragraph(style="List Bullet")
        add_inline(p, line[2:])
        i += 1
        continue

    m = re.match(r"^(\d+)\.\s+(.*)$", line)
    if m:
        p = doc.add_paragraph(style="List Number")
        add_inline(p, m.group(2))
        i += 1
        continue

    if line.strip() == "":
        i += 1
        continue

    # đoạn văn: gộp các dòng liền nhau cho tới khi gặp dòng trống / phần tử khác
    buf = [line.strip()]
    i += 1
    while i < len(lines):
        nxt = lines[i].strip()
        if (not nxt or nxt.startswith(("#", "|", "- ", "```", "---"))
                or re.match(r"^\d+\.\s", nxt)):
            break
        buf.append(nxt)
        i += 1

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    add_inline(p, " ".join(buf))

# --- số trang ở chân trang ---
fp = doc.sections[0].footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run("Trang ")
run.font.name = "Times New Roman"
run.font.size = Pt(10)
fld_run = fp.add_run()
f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"
f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
fld_run._r.append(f1); fld_run._r.append(it); fld_run._r.append(f2)

doc.save(DST)
print("saved", DST)
