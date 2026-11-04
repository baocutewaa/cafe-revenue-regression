import docx
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = docx.Document()

# 1. Enable updateFields in settings
settings_elm = doc.settings._element
update_fields = parse_xml(r'<w:updateFields %s w:val="true"/>' % nsdecls('w'))
settings_elm.append(update_fields)

# 2. Add Heading Title
p = doc.add_paragraph()
p.style = 'TOC Heading' if 'TOC Heading' in doc.styles else 'Heading 1'
p.paragraph_format.space_before = Pt(12)
p.paragraph_format.space_after = Pt(6)
r = p.add_run("MỤC LỤC")
r.bold = True
r.font.name = 'Times New Roman'
r.font.size = Pt(15)

# 3. Add TOC field paragraph
p_toc = doc.add_paragraph()
p_toc.paragraph_format.space_before = Pt(0)
p_toc.paragraph_format.space_after = Pt(0)

# Complex field begin
r1 = p_toc.add_run()
r1._r.append(parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w')))
r2 = p_toc.add_run()
r2._r.append(parse_xml(r'<w:instrText %s xml:space="preserve"> TOC \o "1-3" \h \z \u </w:instrText>' % nsdecls('w')))
r3 = p_toc.add_run()
r3._r.append(parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w')))

# We can also populate initial TOC entries (with PAGEREF and hyperlinks) so that even before Word updates, it displays beautifully!
# And at the end of TOC:
r_end = p_toc.add_run()
r_end._r.append(parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w')))

doc.add_page_break()

# Add headings with Heading 1, 2, 3
h1 = doc.add_paragraph("CHƯƠNG I: GIỚI THIỆU VỀ ĐỀ TÀI", style='Heading 1')
doc.add_paragraph("Nội dung chương 1...")

h2 = doc.add_paragraph("1.1. Lý do chọn đề tài", style='Heading 2')
doc.add_paragraph("Nội dung 1.1...")

doc.save("scratch/test_toc.docx")
print("Saved test_toc.docx successfully!")
