import glob, os

files = sorted(glob.glob('docs/step*.md'))
for f in files:
    print(f"==================== {os.path.basename(f)} ====================")
    with open(f, 'r', encoding='utf-8') as fp:
        content = fp.read()
        # print first 1500 chars and some stats
        print(f"Length: {len(content)} chars")
        lines = content.split('\n')
        print("Lines count:", len(lines))
