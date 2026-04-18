"""Step 1: Replace the F = {...} line in blend.html with new ZEPHYR data."""
import json
from pathlib import Path

HTML = Path("/home/user/bonds/docs/blend.html")
DATA = Path("/home/user/bonds/data/results/blend_factsheet_data.json")

data = json.load(open(DATA))
new_f_line = f"const F = {json.dumps(data, separators=(',', ':'))};\n"

lines = HTML.read_text().splitlines(keepends=True)
# Find the line starting with "const F = "
for i, ln in enumerate(lines):
    if ln.startswith("const F = "):
        print(f"Replacing F constant at line {i+1} (old len={len(ln)}, new len={len(new_f_line)})")
        lines[i] = new_f_line
        break
else:
    raise SystemExit("Could not find 'const F = ' line")

HTML.write_text("".join(lines))
print("Wrote blend.html with new F constant")
