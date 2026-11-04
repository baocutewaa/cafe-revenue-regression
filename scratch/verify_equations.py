import zipfile, re

with zipfile.ZipFile('docs/Group2_Report.docx', 'r') as z:
    doc_xml = z.read('word/document.xml').decode('utf-8')

print("Count of m:oMath tags:", doc_xml.count('<m:oMath>'))
print("Count of m:oMathPara tags:", doc_xml.count('<m:oMathPara>'))

# Check for unrendered $$ or $ in document
raw_double_dollars = re.findall(r'\$\$(.*?)\$\$', doc_xml)
print(f"Remaining raw $$: {len(raw_double_dollars)}")

# Check for raw LaTeX commands in text (like \min, \sum, \frac, \hat)
raw_latex_cmds = re.findall(r'\\(?:min|sum|frac|sqrt|hat|mu|delta|mathbf|mathcal|alpha|beta|lambda|gamma|Omega)\b', doc_xml)
print(f"Remaining raw LaTeX commands in text: {len(raw_latex_cmds)}")
if raw_latex_cmds:
    print("Sample raw latex commands:", raw_latex_cmds[:10])

# Check a sample of m:oMath snippets
omath_samples = re.findall(r'<m:oMath>.*?</m:oMath>', doc_xml)
print(f"\nSample rendered oMath elements count: {len(omath_samples)}")
for i, s in enumerate(omath_samples[:5]):
    # strip XML tags to see text
    text_only = re.sub(r'<.*?>', '', s)
    print(f"oMath {i+1} text content: {text_only}")
