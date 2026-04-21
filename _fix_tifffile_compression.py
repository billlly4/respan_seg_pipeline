"""
Fix tifffile compression API in RESPAN source files.
Old (pre-2022.7.28): compression=('zlib', 1)
New:                 compression='zlib', compressionargs={'level': 1}
"""
import glob, os

files = glob.glob(r'f:/ResPan/cloned_respan/RESPAN/**/*.py', recursive=True)
old = "compression=('zlib', 1)"
new = "compression='zlib', compressionargs={'level': 1}"

total = 0
for f in files:
    txt = open(f, encoding='utf-8').read()
    count = txt.count(old)
    if count:
        open(f, 'w', encoding='utf-8').write(txt.replace(old, new))
        print(f'  {os.path.relpath(f, r"f:/ResPan/cloned_respan")}: {count} replacement(s)')
        total += count

print(f'\nDone — {total} total replacement(s) across {len(files)} files scanned.')
