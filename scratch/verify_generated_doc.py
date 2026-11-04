import docx

doc = docx.Document('docs/Group2_Report.docx')

print(f"Total paragraphs: {len(doc.paragraphs)}")
print(f"Total tables: {len(doc.tables)}")

images_count = 0
for p in doc.paragraphs:
    for r in p.runs:
        if 'drawing' in r._r.xml:
            images_count += 1
print(f"Total inline drawings/images: {images_count}")

# Print all headings
print("\n--- Document Headings Hierarchy ---")
for p in doc.paragraphs:
    if p.text.startswith('CHƯƠNG') or p.text in ['DANH SÁCH THÀNH VIÊN NHÓM 2', 'MỤC LỤC', 'DANH MỤC CÁC TỪ VIẾT TẮT VÀ KÝ HIỆU', 'DANH MỤC BẢNG BIỂU VÀ HÌNH VẼ', 'LỜI NÓI ĐẦU', 'KẾT LUẬN', 'TÀI LIỆU THAM KHẢO', 'PHỤ LỤC']:
        print(f"[H1] {p.text}")
    elif len(p.runs) > 0 and p.runs[0].bold and (p.text.startswith('1.') or p.text.startswith('2.') or p.text.startswith('3.') or p.text.startswith('4.')):
        if len(p.text.split()[0]) <= 5: # e.g. 1.1., 2.1., etc.
            print(f"   [H2] {p.text}")

print("\n--- Tables Check ---")
for i, t in enumerate(doc.tables):
    hdr = [c.text.strip().replace('\n', ' ') for c in t.rows[0].cells]
    print(f"Table {i+1} ({len(t.rows)} rows x {len(t.columns)} cols): {hdr[:3]}")
