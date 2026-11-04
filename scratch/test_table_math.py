import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
import lxml.etree as ET
from latex2mathml.converter import convert as latex2mathml_convert

xslt_path = r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'
xslt = ET.parse(xslt_path)
transform = ET.XSLT(xslt)

def latex_to_omml(latex_code):
    try:
        mathml = latex2mathml_convert(latex_code.strip())
        tree = ET.fromstring(mathml)
        omml = transform(tree)
        return omml.getroot()
    except Exception as e:
        print(f"Error converting '{latex_code}': {e}")
        return None

def fill_paragraph_with_math(p, text, default_font_size=10, bold=False, italic=False, color=None):
    parts = text.split('$')
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 0:
            r = p.add_run(part)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(default_font_size)
            r.bold = bold
            r.italic = italic
            if color:
                r.font.color.rgb = color
        else:
            omml = latex_to_omml(part)
            if omml is not None:
                p._p.append(omml)
            else:
                r = p.add_run(part)
                r.font.name = 'Times New Roman'
                r.font.size = Pt(default_font_size)

doc = docx.Document()
tbl = doc.add_table(rows=3, cols=3)
data = [
    ["$y$", "Giá trị thực tế", "Doanh thu $y \\ge 0$"],
    ["$\\hat{y}$", "Giá trị dự báo", "Doanh thu $\\hat{y} \\ge 0$"],
    ["$\\mu_s$", "Mức nền công suất", "Mức nền tĩnh $\\mu_s$ và $\\delta_{s,t}$"]
]
for row_idx, row in enumerate(data):
    for col_idx, text in enumerate(row):
        cell = tbl.cell(row_idx, col_idx)
        p = cell.paragraphs[0]
        fill_paragraph_with_math(p, text, default_font_size=10)

doc.save("scratch/test_table_math.docx")
print("Saved test_table_math.docx successfully!")
