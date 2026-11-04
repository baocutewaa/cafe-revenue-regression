import docx

doc = docx.Document('docs/Group2_Report.docx')

toc_paras = []
for p in doc.paragraphs:
    if p.style and p.style.name and 'toc' in p.style.name.lower():
        toc_paras.append((p.style.name, p.text))

print(f"Total TOC paragraphs found: {len(toc_paras)}")
for i, (style, text) in enumerate(toc_paras):
    print(f"{i+1:02d} [{style}]: {repr(text)}")
