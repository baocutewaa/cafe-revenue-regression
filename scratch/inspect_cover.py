import docx

doc = docx.Document('docs/Sample_Report.docx')

for i, p in enumerate(doc.paragraphs[:25]):
    has_drawing = 'drawing' in p._element.xml
    print(f"P{i:2d} (align={p.alignment}, style={p.style.name}): {p.text.strip()[:60]} {'[HAS DRAWING]' if has_drawing else ''}")

print("\n--- Tables in first 10 elements ---")
for i, t in enumerate(doc.tables[:3]):
    print(f"Table {i}: {len(t.rows)} rows x {len(t.columns)} cols")
    for r in t.rows:
        print("  ", [c.text.strip().replace('\n', ' ') for c in r.cells])
