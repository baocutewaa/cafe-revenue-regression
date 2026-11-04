import docx

doc = docx.Document('docs/Sample_Report.docx')

print("All Headings in Sample_Report.docx:")
for i, p in enumerate(doc.paragraphs):
    if p.style.name.startswith('Heading') or p.text.isupper() and len(p.text) < 80 and len(p.text) > 3:
        print(f"[{p.style.name}] {p.text}")
