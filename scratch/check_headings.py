import re

with open('scratch/generate_report_full.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Match add_h1, add_h2, add_h3 calls
pattern = re.compile(r'(add_h[123])\((f?["\'])(.*?)\2\)')
matches = pattern.findall(text)

print(f"Total heading calls matched: {len(matches)}")
for fn, _, title in matches:
    print(f"{fn}: {title}")
