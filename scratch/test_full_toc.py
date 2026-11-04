import docx
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = docx.Document()

# Ensure updateFields is in settings
settings_elm = doc.settings._element
if not settings_elm.xpath('w:updateFields'):
    update_fields = parse_xml(r'<w:updateFields %s w:val="true"/>' % nsdecls('w'))
    settings_elm.append(update_fields)

# Title
p_title = doc.add_paragraph()
p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p_title.add_run("MỤC LỤC")
r.bold = True
r.font.name = 'Times New Roman'
r.font.size = Pt(16)

# TOC entries
entries = [
    ("DANH SÁCH THÀNH VIÊN NHÓM 2", "2", 1, "_Toc1001"),
    ("CHƯƠNG I: GIỚI THIỆU VỀ ĐỀ TÀI", "7", 1, "_Toc1002"),
    ("1.1. Lý do chọn đề tài", "7", 2, "_Toc1003"),
    ("1.2. Phạm vi và đối tượng nghiên cứu", "8", 2, "_Toc1004"),
    ("CHƯƠNG II: CƠ SỞ LÝ THUYẾT", "10", 1, "_Toc1005"),
    ("2.1. Tổng quan về bài toán học máy hồi quy chuỗi thời gian dạng bảng", "10", 2, "_Toc1006"),
]

for idx, (title, page, level, tag) in enumerate(entries):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    
    # pStyle
    style_name = f"TOC{level}"
    # Tabs: right dot leader at 9000 dxa (~6.25 inches)
    pPr = p._p.get_or_add_pPr()
    pPr.append(parse_xml(
        f'<w:pStyle {nsdecls("w")} w:val="{style_name}"/>'
    ))
    pPr.append(parse_xml(
        f'<w:tabs {nsdecls("w")}><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>'
    ))
    
    # If first item, prepend TOC field begin & instrText
    if idx == 0:
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="begin"/></w:r>'))
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:instrText xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText></w:r>'))
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="separate"/></w:r>'))
        
    # Hyperlink with PAGEREF
    hl_xml = (
        f'<w:hyperlink {nsdecls("w")} w:anchor="{tag}" w:history="1">'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b w:val="{1 if level==1 else 0}"/><w:sz w:val="{23 if level==1 else 22}"/></w:rPr><w:t>{title}</w:t></w:r>'
        f'<w:r><w:tab/></w:r>'
        f'<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve"> PAGEREF {tag} \\h </w:instrText></w:r>'
        f'<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
        f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b w:val="{1 if level==1 else 0}"/><w:sz w:val="{23 if level==1 else 22}"/></w:rPr><w:t>{page}</w:t></w:r>'
        f'<w:r><w:fldChar w:fldCharType="end"/></w:r>'
        f'</w:hyperlink>'
    )
    p._p.append(parse_xml(hl_xml))
    
    # If last item, append TOC field end
    if idx == len(entries) - 1:
        p._p.append(parse_xml(f'<w:r {nsdecls("w")}><w:fldChar w:fldCharType="end"/></w:r>'))

# Now add sections with bookmarks and Heading styles
def add_heading_with_bookmark(doc, text, level, tag, bookmark_id):
    p = doc.add_paragraph(style=f'Heading {level}')
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    
    # add bookmark
    bm_start = parse_xml(f'<w:bookmarkStart {nsdecls("w")} w:id="{bookmark_id}" w:name="{tag}"/>')
    bm_end = parse_xml(f'<w:bookmarkEnd {nsdecls("w")} w:id="{bookmark_id}"/>')
    p._p.append(bm_start)
    
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.bold = True
    r.font.size = Pt(15 if level==1 else (13.5 if level==2 else 12.5))
    
    p._p.append(bm_end)
    return p

doc.add_page_break()
add_heading_with_bookmark(doc, "DANH SÁCH THÀNH VIÊN NHÓM 2", 1, "_Toc1001", 1)
doc.add_paragraph("Nội dung thành viên...")

doc.add_page_break()
add_heading_with_bookmark(doc, "CHƯƠNG I: GIỚI THIỆU VỀ ĐỀ TÀI", 1, "_Toc1002", 2)
add_heading_with_bookmark(doc, "1.1. Lý do chọn đề tài", 2, "_Toc1003", 3)
doc.add_paragraph("Nội dung 1.1...")

add_heading_with_bookmark(doc, "1.2. Phạm vi và đối tượng nghiên cứu", 2, "_Toc1004", 4)
doc.add_paragraph("Nội dung 1.2...")

doc.add_page_break()
add_heading_with_bookmark(doc, "CHƯƠNG II: CƠ SỞ LÝ THUYẾT", 1, "_Toc1005", 5)
add_heading_with_bookmark(doc, "2.1. Tổng quan về bài toán học máy hồi quy chuỗi thời gian dạng bảng", 2, "_Toc1006", 6)
doc.add_paragraph("Nội dung 2.1...")

doc.save("scratch/test_full_toc.docx")
print("Saved test_full_toc.docx successfully!")
