import re

with open('scratch/generate_report_full.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find all $$ ... $$ and $ ... $
display_math = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)
inline_math = re.findall(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', content)

print(f"Total display math ($$...$$): {len(display_math)}")
for i, m in enumerate(display_math):
    print(f"Display {i+1}: {m.strip()}")

print(f"\nTotal inline math ($...$): {len(inline_math)}")
sample_inline = set(inline_math)
for m in sorted(sample_inline)[:30]:
    print(f"Inline: {m.strip()}")
