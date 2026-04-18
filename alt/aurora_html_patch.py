"""Inject aurora_factsheet_data.json into docs/aurora.html."""
import json
from pathlib import Path

ROOT = Path("/home/user/bonds")
HTML = ROOT / "docs/aurora.html"
DATA = ROOT / "data/results/aurora_factsheet_data.json"

data = json.loads(DATA.read_text())
html = HTML.read_text()
lines = html.split("\n")

new_f_line = f"const F = {json.dumps(data, separators=(',', ':'))};"
for i, ln in enumerate(lines):
    if ln.startswith("const F = "):
        print(f"Replacing F at line {i+1} (old len={len(ln)}, new len={len(new_f_line)})")
        lines[i] = new_f_line
        break
else:
    raise SystemExit("Did not find `const F = ` line in aurora.html")

HTML.write_text("\n".join(lines))
print(f"Wrote {HTML}")
