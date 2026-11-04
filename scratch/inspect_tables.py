import docx

doc = docx.Document('docs/Sample_Report.docx')

for i, table in enumerate(doc.tables):
    print(f"\n--- Table {i} (style={table.style.name if table.style else 'None'}, rows={len(table.rows)}, cols={len(table.columns)}) ---")
    header = [c.text.strip().replace('\n', ' ') for c in table.rows[0].cells]
    print("Header:", header)
    if len(table.rows) > 1:
        row1 = [c.text.strip().replace('\n', ' ') for c in table.rows[1].cells]
        print("Row 1:", row1)
