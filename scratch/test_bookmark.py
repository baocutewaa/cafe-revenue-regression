import docx
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

doc = docx.Document()
p = doc.add_paragraph(style='Heading 1')
bm_id = 1
tag = '_Toc1001'
p._p.append(parse_xml(f'<w:bookmarkStart {nsdecls("w")} w:id="{bm_id}" w:name="{tag}"/>'))
r = p.add_run('CHƯƠNG I: GIỚI THIỆU')
p._p.append(parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bm_id}"/>'))

print(p._p.xml)
