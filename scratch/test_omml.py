import os
import lxml.etree as ET
import latex2mathml.commands
from latex2mathml.converter import convert as latex2mathml_convert
from docx import Document
from docx.oxml import parse_xml

xslt_path = r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'
xslt = ET.parse(xslt_path)
transform = ET.XSLT(xslt)

latex_expr = r"\text{RMSLE}(y, \hat{y}) = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \ln(1 + \max(0, \hat{y}_i)) - \ln(1 + y_i) \right)^2}"

mathml = latex2mathml_convert(latex_expr)
mathml_tree = ET.fromstring(mathml)
omml = transform(mathml_tree)

print("Transformed OMML root tag:", omml.getroot().tag)
print("OMML XML snippet:", ET.tostring(omml, pretty_print=True).decode('utf-8')[:400])

doc = Document()
p = doc.add_paragraph("Đây là phương trình chuẩn Word:")
# In docx, add omml root directly to paragraph XML
p._p.append(omml.getroot())
doc.save("scratch/test_math.docx")
print("Saved test_math.docx successfully!")
