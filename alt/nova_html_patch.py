"""Inject nova_factsheet_data.json into docs/nova.html AND adjust
page metadata (title, nav bar active tab, header descriptor). Idempotent."""
import json
import re
from pathlib import Path

ROOT = Path("/home/user/bonds")
HTML = ROOT / "docs/nova.html"
DATA = ROOT / "data/results/nova_factsheet_data.json"

data = json.loads(DATA.read_text())
html = HTML.read_text()

# 1) Replace <title>
html = re.sub(
    r"<title>.*?</title>",
    "<title>NOVA — Max-Growth Momentum Factsheet</title>",
    html, count=1, flags=re.S,
)

# 2) Replace H1 header text and subtitle (matches aurora's header block)
html = html.replace(
    "<h1>AURORA — Diversified Growth Strategy</h1>",
    "<h1>NOVA — Max-Growth Momentum</h1>",
)
html = html.replace(
    '<div class="sub">3-Sleeve Income + Momentum + Managed-Futures | Weekly Momo Rebal | Targets 20%+ Return</div>',
    '<div class="sub">Unified Cross-Sectional Momentum on 3x ETFs + BTC + ETH | Weekly Rebal | Targets 50%+ Return</div>',
)

# 3) Nav bar: make NOVA the active one on this page, demote AURORA back to inactive
#    Add NOVA tab if missing. Convert AURORA's active-tab style to inactive.
INACTIVE = ('padding:6px 16px;border-radius:20px;background:var(--card);color:var(--t1);'
            'text-decoration:none;font-size:0.82rem;font-weight:500;border:1px solid var(--border)')
ACTIVE = ('padding:6px 16px;border-radius:20px;background:var(--accent);color:#fff;'
          'text-decoration:none;font-size:0.82rem;font-weight:600;border:1px solid var(--accent)')

# Demote aurora's current active → inactive on this page
html = html.replace(
    f'<a href="aurora.html" style="{ACTIVE}">AURORA</a>',
    f'<a href="aurora.html" style="{INACTIVE}">AURORA</a>',
)

# Add NOVA active link after aurora if not already present
if "nova.html" not in html:
    html = html.replace(
        f'<a href="aurora.html" style="{INACTIVE}">AURORA</a>',
        f'<a href="aurora.html" style="{INACTIVE}">AURORA</a>\n'
        f'<a href="nova.html" style="{ACTIVE}">NOVA</a>',
    )
else:
    # Ensure NOVA is active on the NOVA page
    html = html.replace(
        f'<a href="nova.html" style="{INACTIVE}">NOVA</a>',
        f'<a href="nova.html" style="{ACTIVE}">NOVA</a>',
    )

# 4) Inject JSON
lines = html.split("\n")
new_f_line = f"const F = {json.dumps(data, separators=(',', ':'))};"
for i, ln in enumerate(lines):
    if ln.startswith("const F = "):
        print(f"Replacing F at line {i+1} (new len={len(new_f_line)})")
        lines[i] = new_f_line
        break
else:
    raise SystemExit("Did not find `const F = ` line in nova.html")

HTML.write_text("\n".join(lines))
print(f"Wrote {HTML}")
