import docx
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

doc = docx.Document()
section = doc.sections[0]
sectPr = section._sectPr
pgBorders = parse_xml(
    f'<w:pgBorders {nsdecls("w")} w:display="firstPage" w:offsetFrom="page">'
    f'<w:top w:val="double" w:sz="12" w:space="24" w:color="1B365D"/>'
    f'<w:left w:val="double" w:sz="12" w:space="24" w:color="1B365D"/>'
    f'<w:bottom w:val="double" w:sz="12" w:space="24" w:color="1B365D"/>'
    f'<w:right w:val="double" w:sz="12" w:space="24" w:color="1B365D"/>'
    f'</w:pgBorders>'
)
sectPr.append(pgBorders)
doc.add_paragraph("Test cover page border")
doc.save("scratch/test_border.docx")
print("Saved test_border.docx successfully!")
