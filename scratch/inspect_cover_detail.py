import docx

doc = docx.Document('docs/Sample_Report.docx')

print("Paragraphs 0 to 15 XML snippets:")
for i in range(16):
    p = doc.paragraphs[i]
    print(f"--- P {i} (text: '{p.text}') ---")
    for r in p.runs:
        if r._element.xpath('.//a:blip'):
            for blip in r._element.xpath('.//a:blip'):
                print("   Blip embed:", blip.attrib)
        if 'w:br' in r._element.xml:
            print("   Contains break:", [b.attrib for b in r._element.xpath('.//w:br')])
