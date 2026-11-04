import zipfile, re

with zipfile.ZipFile('docs/Sample_Report.docx', 'r') as z:
    doc_xml = z.read('word/document.xml').decode('utf-8')

# Find the section containing TOC
idx_toc_begin = doc_xml.find('TOC \\o')
if idx_toc_begin != -1:
    snippet = doc_xml[idx_toc_begin - 150 : idx_toc_begin + 2500]
    print("--- TOC Snippet from Sample_Report.docx ---")
    print(snippet)
