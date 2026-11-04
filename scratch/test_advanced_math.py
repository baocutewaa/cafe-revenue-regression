import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import lxml.etree as ET
from latex2mathml.converter import convert as latex2mathml_convert

xslt_path = r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'
xslt = ET.parse(xslt_path)
transform = ET.XSLT(xslt)

def latex_to_omml(latex_code):
    """Converts a LaTeX math string to an lxml Element for oMath."""
    # Clean some LaTeX commands that latex2mathml might struggle with or needs clean syntax
    code = latex_code.strip()
    # Replace \mathbb{R} with standard R if needed, or keep
    try:
        mathml = latex2mathml_convert(code)
        tree = ET.fromstring(mathml)
        omml = transform(tree)
        root = omml.getroot()
        # root is <m:oMath>
        return root
    except Exception as e:
        print(f"Error converting '{code}': {e}")
        return None

def add_display_equation(doc, latex_code):
    """Adds a centered display equation paragraph to doc."""
    omml_elem = latex_to_omml(latex_code)
    if omml_elem is not None:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        
        # Create m:oMathPara wrapping the oMath
        m_ns = "http://schemas.openxmlformats.org/officeDocument/2006/math"
        oMathPara = ET.Element(f"{{{m_ns}}}oMathPara")
        oMathPara.append(omml_elem)
        p._p.append(oMathPara)
        return p
    else:
        # Fallback to plain text
        p = doc.add_paragraph(latex_code)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return p

def add_formatted_p(doc, text, bold=False, italic=False, size=12, color=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before=0, space_after=4, line_spacing=1.25):
    """Adds a paragraph that parses $inline_math$ and renders it with native oMath."""
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    
    parts = text.split('$')
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 0:
            # Regular text
            r = p.add_run(part)
            r.font.name = 'Times New Roman'
            r.font.size = Pt(size)
            r.bold = bold
            r.italic = italic
            if color:
                r.font.color.rgb = color
        else:
            # Inline math!
            omml = latex_to_omml(part)
            if omml is not None:
                p._p.append(omml)
            else:
                r = p.add_run(f"${part}$")
                r.font.name = 'Times New Roman'
                r.font.size = Pt(size)
    return p

doc = docx.Document()
add_formatted_p(doc, "Doanh thu dự báo $\\hat{y}$ được phân rã thành mức nền $\\mu_s$ và phần dư $\\delta_{s,t}$.")
add_display_equation(doc, r"\text{RMSLE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \ln(1 + \max(0, \hat{y}_i)) - \ln(1 + y_i) \right)^2}")
add_formatted_p(doc, "Mô hình đạt hệ số xác định $R^2 = 0.9055$ và độ dốc xu thế $\\beta = 0.000318$.")

doc.save("scratch/test_math_advanced.docx")
print("Saved test_math_advanced.docx successfully!")
