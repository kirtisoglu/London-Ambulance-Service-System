"""Report blank space per page of a compiled PDF.

For every page, the blank at the bottom of the text block and the largest
blank band inside it, as fractions of the text block. Pages above the two
thresholds are listed; only the last page should be.

    python paper/text_generators/check_page_gaps.py main_arxiv.pdf /tmp/pages 0.05 0.07
"""
import subprocess, sys, glob, os
import numpy as np
from PIL import Image
pdf, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for f in glob.glob(out + "/p-*.pgm"): os.remove(f)
subprocess.run(["pdftoppm", "-gray", "-r", "36", pdf, out + "/p"], check=True)
rows = []
for i, f in enumerate(sorted(glob.glob(out + "/p-*.pgm")), start=1):
    a = np.array(Image.open(f))
    h, w = a.shape
    top, bot = int(h * 1.0 / 11), int(h * 10.0 / 11)        # text block (1in margins); the page number sits below it
    x0, x1 = int(w * 1.2 / 8.5), int(w * 7.5 / 8.5)         # skip the line numbers in the margin
    ink = (a[top:bot, x0:x1] < 200).any(axis=1)
    n = len(ink)
    idx = np.where(ink)[0]
    if len(idx) == 0:
        rows.append((i, 1.0, 1.0)); continue
    bottom = (n - 1 - idx[-1]) / n
    gaps = np.diff(idx)
    internal = (gaps.max() - 1) / n if len(gaps) else 0.0
    rows.append((i, bottom, internal))
bad = [(i, b, g) for i, b, g in rows if b > float(sys.argv[3]) or g > float(sys.argv[4])]
print(f"{len(rows)} pages; pages with a bottom blank or internal gap above the thresholds of the text block:")
for i, b, g in bad:
    print(f"  p{i}: bottom {b:.0%}, internal {g:.0%}")
