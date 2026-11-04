import docx
import html
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE

doc = docx.Document()

# Add TOC styles
for lvl in range(1, 4):
    name = f'TOC {lvl}'
    s = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    s.font.name = 'Times New Roman'
    s.font.size = Pt(11)
    if lvl == 1:
        s.font.bold = True
        s.paragraph_format.left_indent = Inches(0)
    elif lvl == 2:
        s.paragraph_format.left_indent = Inches(0.2)
    elif lvl == 3:
        s.paragraph_format.left_indent = Inches(0.4)

# Heading styles
h1 = doc.styles['Heading 1']
h1.font.name = 'Times New Roman'
h1.font.size = Pt(15)
h1.font.bold = True

h2 = doc.styles['Heading 2']
h2.font.name = 'Times New Roman'
h2.font.size = Pt(13)
h2.font.bold = True

h3 = doc.styles['Heading 3']
h3.font.name = 'Times New Roman'
h3.font.size = Pt(12)
h3.font.bold = True

# updateFields
settings_elm = doc.settings._element
if not settings_elm.xpath('w:updateFields'):
    settings_elm.append(parse_xml(r'<w:updateFields %s w:val="true"/>' % nsdecls('w')))

p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p_title.add_run("MỤC LỤC")
r.bold = True
r.font.name = 'Times New Roman'
r.font.size = Pt(16)

# TOC entries
entries = [
    ("CHƯƠNG I: GIỚI THIỆU VỀ ĐỀ TÀI", "2", 1, "_Toc1001"),
    ("1.1. Lý do chọn đề tài", "2", 2, "_Toc1002"),
    ("1.1.1. Bối cảnh thị trường F&B", "2", 3, "_Toc1003"),
    ("CHƯƠNG II: CƠ SỞ LÝ THUYẾT", "3", 1, "_Toc1004"),
]

for idx, (title, page, level, tag) in enumerate(entries):
    p = doc.add_paragraph(style=f'TOC {level}')
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    
    pPr = p._p.get_or_add_pPr()
    pPr.append(parse_xml(
        f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>'
    ))
    
    if idx == 0:
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="begin"/></w:r>'))
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'))
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="separate"/></w:r>'))
        
    title_esc = html.escape(title)
    hl_xml = (
        f'<w:hyperlink {nsdecls("w")} w:anchor="{tag}" w:history="1">'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b w:val="{1 if level==1 else 0}"/><w:sz w:val="{22}"/></w:rPr><w:t>{title_esc}</w:t></w:r>'
        f'<w:r><w:tab/></w:r>'
        f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> PAGEREF {tag} \\h </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b w:val="{1 if level==1 else 0}"/><w:sz w:val="{22}"/></w:rPr><w:t>{page}</w:t></w:r>'
        f'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        f'</w:hyperlink>'
    )
    p._p.append(parse_xml(hl_xml))
    
    if idx == len(entries) - 1:
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="end"/></w:r>'))

def add_heading(doc, text, level, tag, bmid):
    p = doc.add_paragraph(style=f'Heading {level}')
    bm_start = parse_xml(f'<w:bookmarkStart {nsdecls("w")} w:id="{bmid}" w:name="{tag}"/>')
    bm_end = parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bmid}"/>')
    p._p.append(bm_start)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    p._p.append(bm_end)

doc.add_page_break()
add_heading(doc, "CHƯƠNG I: GIỚI THIỆU VỀ ĐỀ TÀI", 1, "_Toc1001", 1)
add_heading(doc, "1.1. Lý do chọn đề tài", 2, "_Toc1002", 2)
add_heading(doc, "1.1.1. Bối cảnh thị trường F&B", 3, "_Toc1003", 3)
doc.add_paragraph("Nội dung chương 1...")

doc.add_page_break()
add_heading(doc, "CHƯƠNG II: CƠ SỞ LÝ THUYẾT", 1, "_Toc1004", 4)
doc.add_paragraph("Nội dung chương 2...")

doc.save("scratch/test_toc_indent.docx")
print("Saved test_toc_indent.docx successfully!")
