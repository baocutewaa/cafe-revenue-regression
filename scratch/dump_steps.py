import glob, os

files = sorted(glob.glob('docs/step*.md'))
for f in files:
    print(f"\n############################################################")
    print(f"##### {os.path.basename(f)} #####")
    print(f"############################################################\n")
    with open(f, 'r', encoding='utf-8') as fp:
        print(fp.read())
