import json
D=json.load(open("/home/user/_wsdata.json"))
DATA=json.dumps(D,separators=(",",":"))
PICKS=json.dumps(json.load(open("/home/user/_picks.json")),separators=(",",":"))
html=r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>WAVE & SUMMIT — Strategy Brief</title>
<style>
:root{--bg:#fff;--txt:#111418;--mut:#6b7280;--line:#e5e7eb;--card:#fafafa;--good:#15803d;--bad:#b91c1c}
*{box-sizing:border-box;-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--txt);font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding:0 0 56px}
.wrap{max-width:660px;margin:0 auto;padding:0 16px}
header{padding:26px 16px 12px;text-align:center;border-bottom:1px solid var(--line)}
header h1{font-size:25px;margin:0 0 4px;letter-spacing:.5px;font-weight:800}
header p{color:var(--mut);font-size:13px;margin:0}
nav{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);backdrop-filter:blur(6px);border-bottom:1px solid var(--line);display:flex;gap:8px;padding:10px 16px}
nav button{flex:1;text-align:center;color:var(--txt);padding:11px 6px;border-radius:10px;font-weight:700;font-size:15px;background:#fff;border:1.5px solid var(--txt);cursor:pointer}
nav button.on{background:var(--txt);color:#fff}
section{padding-top:16px}
.badge{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.6px;padding:4px 10px;border-radius:999px;text-transform:uppercase;border:1.5px solid var(--txt)}
h2{font-size:23px;margin:9px 0 2px;font-weight:800;letter-spacing:.5px}
.sub{color:var(--mut);font-size:13.5px;margin:0 0 12px}
.chips{display:flex;flex-wrap:wrap;gap:7px;margin:0 0 12px}
.chips button{font-size:12.5px;font-weight:600;color:var(--txt);background:#fff;border:1px solid var(--line);border-radius:999px;padding:6px 12px;cursor:pointer}
.chips button.on{background:var(--txt);color:#fff;border-color:var(--txt)}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin:0 0 14px}
.card h3{margin:0 0 11px;font-size:13px;color:var(--mut);font-weight:700;text-transform:uppercase;letter-spacing:.5px}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}
.m{background:#fff;border:1px solid var(--line);border-radius:11px;padding:12px 8px;text-align:center}
.m .v{font-size:21px;font-weight:800;line-height:1.1}
.m .l{font-size:11px;color:var(--mut);margin-top:3px}
.m .q{font-size:10.5px;color:var(--mut);margin-top:2px}
.good{color:var(--good)}.bad{color:var(--bad)}
svg{width:100%;height:auto;display:block}
.leg{display:flex;gap:16px;justify-content:center;font-size:12px;color:var(--mut);margin-top:8px}
.leg i{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:8px 6px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.4px;cursor:pointer;user-select:none}
th.s::after{content:" ▾";opacity:.6}th.s.asc::after{content:" ▴"}
ul{margin:6px 0 0;padding-left:18px}li{margin:5px 0;font-size:14.5px}ul.tight li{margin:3px 0}
.row{display:flex;justify-content:space-between;gap:10px;padding:9px 0;border-bottom:1px solid var(--line);font-size:14px}
.row:last-child{border-bottom:none}.row .k{color:var(--mut);white-space:nowrap}.row .val{font-weight:600;text-align:right}
.pill{font-size:12px;color:var(--mut);background:#fff;border:1px solid var(--line);border-radius:8px;padding:3px 9px;display:inline-block;margin:3px 4px 0 0}
.note{font-size:13px;background:#f3f4f6;border-left:3px solid var(--txt);border-radius:6px;padding:10px 12px;margin-top:4px}
footer{color:var(--mut);font-size:12px;text-align:center;padding:20px 16px 0;line-height:1.7;border-top:1px solid var(--line);margin-top:24px}
.hide{display:none}
</style></head><body>
<header><h1>WAVE &amp; SUMMIT</h1><p>Equity strategy brief · survivorship-clean point-in-time · interactive</p></header>
<nav class="wrap"><button id="tab-wave" class="on" onclick="setStrat('wave')">WAVE</button><button id="tab-summit" onclick="setStrat('summit')">SUMMIT</button></nav>
<div class="wrap">
  <span class="badge" id="badge"></span>
  <h2 id="title"></h2>
  <p class="sub" id="subtitle"></p>

  <div class="chips" id="chips"></div>

  <div class="card"><h3 id="methdr"></h3><div class="metrics" id="metrics"></div><p class="note" id="metnote"></p></div>

  <div class="card"><h3>Growth of $1 — <span id="charthdr"></span></h3>
    <div id="chart"></div>
    <div class="leg"><span><i style="background:#111418"></i><b id="lgS">Strategy</b></span><span><i style="background:#9ca3af"></i>QQQ</span></div>
  </div>

  <div class="card" id="curCard"><h3 id="curHdr"></h3><div id="curBody"></div>
    <p style="font-size:12px;color:var(--mut);margin:10px 0 0" id="curNote"></p></div>

  <div class="card" id="histCard"><h3>Notable historical winners <span style="text-transform:none;font-weight:400">(tap to sort)</span></h3>
    <table id="hist"><thead><tr>
      <th data-k="t">Ticker</th><th data-k="entry">Entry</th><th data-k="held">Mo held</th><th data-k="ret" class="s">Return</th>
    </tr></thead><tbody></tbody></table>
    <p style="font-size:12px;color:var(--mut);margin:10px 0 0">Actual backtest trades (2015–25), ranked by realized return. The fat tail that drives the strategy.</p></div>

  <div class="card"><h3>Annual returns <span style="text-transform:none;font-weight:400">(tap a header to sort)</span></h3>
    <table id="annual"><thead><tr>
      <th data-k="year" class="s asc">Year</th><th data-k="strat" id="thS">Strat</th><th data-k="qqq">QQQ</th><th data-k="exc">Excess</th>
    </tr></thead><tbody></tbody></table>
  </div>

  <div class="card" id="factorCard"><h3>Factor long/short sleeves <span style="text-transform:none;font-weight:400">(tap to sort)</span></h3>
    <table id="factors"><thead><tr>
      <th data-k="name">Factor</th><th data-k="sharpe" class="s">Sharpe</th><th data-k="ann">Ann %</th>
    </tr></thead><tbody></tbody></table>
    <p style="font-size:12px;color:var(--mut);margin:10px 0 0">Gross dollar-neutral L/S Sharpe per signal. Value (B/M) leads; combined book reaches the headline above.</p>
  </div>

  <div id="info"></div>
  <footer>WAVE = deploy now (long-only). SUMMIT = bigger alpha, when you can short.<br>Survivorship-clean &amp; point-in-time; null-gauntlet validated. Research, not advice.</footer>
</div>
<script>
const D=__DATA__;const PK=__PICKS__;
const CFG={
 wave:{name:"WAVE",badge:"Deploy now · long-only · no margin",sub:"ML stock-picker — ride winners, cut losers. ~12 names, monthly.",factors:false,
  info:`<div class="card"><h3>The edge — why it works</h3><ul class="tight">
   <li>Win rate just <b>53%</b> — the edge is <b>not</b> hit-rate.</li>
   <li>Avg <span class="good">win +45%</span> vs avg <span class="bad">loss −8%</span> (≈ 6 : 1).</li>
   <li>Winners held <b>15.7 months</b>, losers cut in <b>3.1 months</b>.</li>
   <li>Top 5% of trades drive <b>42%</b> of all profit (fat-tailed).</li></ul></div>
   <div class="card"><h3>The rules</h3>
   <div class="row"><span class="k">Universe</span><span class="val">US stocks ≥ $3, above 10-mo MA</span></div>
   <div class="row"><span class="k">Select</span><span class="val">Top ~12 by ML score<br>+ momentum &gt; 0 (runner-gate)<br>+ ML score rising (acceleration)</span></div>
   <div class="row"><span class="k">Size</span><span class="val">Equal at entry, then ride</span></div>
   <div class="row"><span class="k">Cut losers</span><span class="val">−30% trailing stop, or<br>close below 10-mo MA</span></div>
   <div class="row"><span class="k">Rebalance</span><span class="val">Monthly; refill to ~12</span></div></div>
   <div class="card"><h3>Tested — did <span class="bad">not</span> help</h3>
   <span class="pill">ensembles</span><span class="pill">feature selection</span><span class="pill">extra gates</span><span class="pill">vol-sizing</span><span class="pill">regime timing</span><span class="pill">volume/TA features</span><span class="pill">“banger” target</span><span class="pill">turnaround sleeve</span><span class="pill">score-blending</span></div>
   <div class="card"><h3>Honest caveats</h3><ul class="tight">
   <li>Small/mid-cap tilt → capacity-limited.</li><li>~Monthly turnover (costs apply).</li>
   <li>Close-based prices; ML needs history (2015+).</li>
   <li>A diversified blend did <b>not</b> beat WAVE alone (sleeves ~0.55 correlated).</li></ul></div>`},
 summit:{name:"SUMMIT",badge:"Archived · needs shorting + leverage",sub:"Market-neutral long/short — pure alpha, ~0 correlation to the market.",factors:true,
  info:`<div class="card"><h3>The two engines</h3>
   <div class="row"><span class="k">Linear factor L/S</span><span class="val">value, gross-profitability,<br>Piotroski, momentum, quality</span></div>
   <div class="row"><span class="k">ML L/S</span><span class="val">36-feature gradient boosting</span></div>
   <div class="row"><span class="k">Best single factor</span><span class="val">Value (B/M) — Sharpe 1.38</span></div>
   <div class="row"><span class="k">Construction</span><span class="val">Beta-neutral; long top decile,<br>short bottom decile (≥$10 mcap)</span></div></div>
   <div class="card"><h3>Risk-neutral construction</h3>
   <p style="font-size:12.5px;color:var(--mut);margin:0 0 8px">Two design choices make it genuinely market-neutral and deployable: <b>borrow-aware shorts</b> — short only <b>≥$10 mcap</b> names (reliably borrowable; micro-cap shorts add squeeze risk, not edge); and <b>beta-neutral</b> sizing — scale the short leg to zero <i>net beta</i> (not just dollar-neutral), removing the residual market exposure that drives drawdowns. The metrics below are <b>all-in net</b> of every modeled cost.</p>
   <div class="metrics">
   <div class="m"><div class="v">2.42</div><div class="l">net Sharpe</div><div class="q">all-in</div></div>
   <div class="m"><div class="v">−17%</div><div class="l">max drawdown</div><div class="q">~16% vol</div></div>
   <div class="m"><div class="v">0.00</div><div class="l">corr to QQQ</div><div class="q">market-neutral</div></div></div>
   <p style="font-size:12px;color:var(--mut);margin:8px 0 0">The edge is the ML cross-sectional signal. Tested &amp; rejected as added alpha: sector- &amp; size-neutral, ML+linear ensemble, residual-momentum, model-averaging, alternative ML models, short stop-losses, return-tranching — each shown not to help (the model already subsumes public-data factors).</p></div>
   <div class="card"><h3>Costs &amp; capacity — fully modeled</h3>
   <p style="font-size:12.5px;color:var(--mut);margin:0 0 8px">Net of <b>tiered spread</b> (4–40bps by size), <b>square-root market impact</b>, <b>tiered borrow</b> (1–6%/yr), <b>financing</b>, and a <b>delisting stress</b>. Dividends-on-shorts are already in total-return prices.</p>
   <div class="row"><span class="k">Gross → all-in net Sharpe</span><span class="val">2.62 → <b>2.42</b> (costs ≈ 0.2)</span></div>
   <div class="row"><span class="k">Capacity</span><span class="val">Sharpe 2.42 @ $100M;<br><b>2.35 @ $1B</b> (&lt;1% of ADV)</span></div>
   <div class="row"><span class="k">Turnover</span><span class="val">~61%/mo; quarterly + 2× buffer</span></div></div>
   <div class="card"><h3>Portable alpha — overlay on QQQ</h3><div class="metrics">
   <div class="m"><div class="v">2.41</div><div class="l">QQQ + 1× α</div><div class="q">DD −23%</div></div>
   <div class="m"><div class="v">2.41</div><div class="l">½QQQ+½+α</div><div class="q">DD −12%</div></div>
   <div class="m"><div class="v">2.62</div><div class="l">QQQ + 2× α</div><div class="q">DD −30%</div></div></div>
   <p style="font-size:12.5px;color:var(--mut);margin:10px 0 0">~0-correlated alpha lifts a QQQ core above either alone. All-in net; overlay needs margin.</p></div>
   <div class="card"><h3>Honest caveats</h3><ul class="tight">
   <li>Requires <b>shorting</b> (≥$10 mcap, borrowable) and <b>leverage/margin</b> for the overlay.</li>
   <li>Pure L/S max drawdown ≈ −17%; native vol ~16%/yr (can be levered to dial return).</li>
   <li>ADV/impact use a <b>mcap×turnover proxy</b> (no tick-volume data); beta from trailing 12-mo returns.</li>
   <li>Holdout was evaluated repeatedly across the search — treat the exact Sharpe as an <b>optimistic</b> point estimate; the parameter surface is stable (2.3–2.5).</li>
   <li><b>Outside the long-only mandate</b> — parked until shorting is allowed.</li></ul></div>`}
};
const PERIODS=[["All","2015-01","2025-12"],["2015–18","2015-01","2018-12"],["2019–21","2019-01","2021-12"],["2022–25","2022-01","2025-12"],["2023–25","2023-01","2025-12"]];
let S="wave",P=0,sortA={k:"year",asc:true},sortF={k:"sharpe",asc:false},sortH={k:"ret",asc:false};
const $=id=>document.getElementById(id);
function rng(){const[_,a,b]=PERIODS[P];const i0=D.dates.indexOf(a),i1=D.dates.indexOf(b);return[i0,i1<0?D.dates.length-1:i1];}
function slice(key){const[i0,i1]=rng();let xs=[],r=[];for(let i=i0;i<=i1;i++){const v=D[key][i],q=D.qqq[i];if(v==null)continue;xs.push(D.dates[i]);r.push(v);}return{xs,r};}
function pair(){const[i0,i1]=rng();let xs=[],s=[],q=[];for(let i=i0;i<=i1;i++){if(D[S][i]==null||D.qqq[i]==null)continue;xs.push(D.dates[i]);s.push(D[S][i]);q.push(D.qqq[i]);}return{xs,s,q};}
function met(r){if(r.length<3)return{c:NaN,sh:NaN,dd:NaN};let g=1,eq=[],pk=-1e9,dd=0,m=0;for(const x of r){g*=1+x;eq.push(g);}m=r.reduce((a,b)=>a+b,0)/r.length;let sd=Math.sqrt(r.reduce((a,b)=>a+(b-m)**2,0)/(r.length-1));pk=-1e9;for(const e of eq){if(e>pk)pk=e;dd=Math.min(dd,e/pk-1);}return{c:Math.pow(g,12/r.length)-1,sh:sd?m/sd*Math.sqrt(12):NaN,dd};}
function pct(x){return(x>=0?"+":"")+(x*100).toFixed(1)+"%";}
function f2(x){return x.toFixed(2);}
function renderMetrics(){const{s,q}=pair();const a=met(s),b=met(q);
 $("methdr").textContent=CFG[S].name+" vs QQQ — "+PERIODS[P][0];
 const cls=(x,y)=>x>=y?"good":"bad";const ddcls=(x,y)=>x>=y?"good":"bad";
 $("metrics").innerHTML=
  `<div class="m"><div class="v ${cls(a.c,b.c)}">${pct(a.c)}</div><div class="l">CAGR</div><div class="q">QQQ ${pct(b.c)}</div></div>
   <div class="m"><div class="v ${cls(a.sh,b.sh)}">${f2(a.sh)}</div><div class="l">Sharpe</div><div class="q">QQQ ${f2(b.sh)}</div></div>
   <div class="m"><div class="v ${ddcls(a.dd,b.dd)}">${pct(a.dd)}</div><div class="l">max drawdown</div><div class="q">QQQ ${pct(b.dd)}</div></div>`;
 $("metnote").textContent = S=="wave"
   ? "Survivorship-clean; delisted names included. Metrics recompute for the selected period."
   : "Market-neutral net of ~6%/yr borrow. Gross/overlay Sharpe higher (see below).";
}
function chart(){const{xs,s,q}=pair();if(s.length<2){$("chart").innerHTML="";return;}
 let es=[1],eq=[1];for(let i=0;i<s.length;i++){es.push(es[es.length-1]*(1+s[i]));eq.push(eq[eq.length-1]*(1+q[i]));}
 es=es.slice(1);eq=eq.slice(1);
 const W=360,H=190,pad=6,bot=16;const all=es.concat(eq);let lo=Math.min(...all),hi=Math.max(...all);
 const ly=v=>{const t=(Math.log(v)-Math.log(lo))/(Math.log(hi)-Math.log(lo)||1);return pad+(H-pad-bot)*(1-t);};
 const lx=i=>pad+(W-2*pad)*(i/(es.length-1||1));
 const path=a=>a.map((v,i)=>(i?"L":"M")+lx(i).toFixed(1)+" "+ly(v).toFixed(1)).join(" ");
 const yr=xs.map(d=>d.slice(0,4));const ticks=[];let last="";xs.forEach((d,i)=>{const y=d.slice(0,4);if(y!=last){ticks.push([i,y]);last=y;}});
 let g="";ticks.forEach(([i,y])=>{const x=lx(i).toFixed(1);g+=`<line x1="${x}" y1="${pad}" x2="${x}" y2="${H-bot}" stroke="#eee"/><text x="${x}" y="${H-4}" font-size="9" fill="#9ca3af" text-anchor="middle">'${y.slice(2)}</text>`;});
 $("charthdr").textContent=PERIODS[P][0]+" ("+xs[0]+" → "+xs[xs.length-1]+")";
 $("chart").innerHTML=`<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:200px">${g}
   <path d="${path(eq)}" fill="none" stroke="#9ca3af" stroke-width="1.6"/>
   <path d="${path(es)}" fill="none" stroke="#111418" stroke-width="2.4"/>
   <text x="${(W-pad)}" y="${ly(es[es.length-1])-4}" font-size="10" font-weight="700" fill="#111418" text-anchor="end">${es[es.length-1].toFixed(2)}×</text>
   <text x="${(W-pad)}" y="${ly(eq[eq.length-1])+11}" font-size="10" fill="#6b7280" text-anchor="end">${eq[eq.length-1].toFixed(2)}×</text></svg>`;
}
function annualRows(){const yrs={};for(let i=0;i<D.dates.length;i++){const y=D.dates[i].slice(0,4);const sv=D[S][i],qv=D.qqq[i];if(!yrs[y])yrs[y]={s:1,q:1,hs:false,hq:false};if(sv!=null){yrs[y].s*=1+sv;yrs[y].hs=true;}if(qv!=null){yrs[y].q*=1+qv;yrs[y].hq=true;}}
 let rows=[];for(const y in yrs){if(!yrs[y].hs)continue;const sr=yrs[y].s-1,qr=yrs[y].q-1;rows.push({year:y,strat:sr,qqq:qr,exc:sr-qr});}return rows;}
function renderAnnual(){let rows=annualRows();const k=sortA.k;rows.sort((a,b)=>{let x=a[k],y=b[k];if(k=="year"){x=+x;y=+y;}return sortA.asc?x-y:y-x;});
 $("thS").textContent=CFG[S].name;
 document.querySelectorAll("#annual th").forEach(th=>{th.classList.toggle("s",th.dataset.k==k);th.classList.toggle("asc",th.dataset.k==k&&sortA.asc);});
 const tb=$("annual").querySelector("tbody");tb.innerHTML=rows.map(r=>`<tr><td>${r.year}</td><td class="${r.strat>=0?'good':'bad'}">${pct(r.strat)}</td><td>${pct(r.qqq)}</td><td class="${r.exc>=0?'good':'bad'}">${pct(r.exc)}</td></tr>`).join("");}
function renderFactors(){const card=$("factorCard");if(!CFG[S].factors){card.classList.add("hide");return;}card.classList.remove("hide");
 let rows=D.factors.slice();const k=sortF.k;rows.sort((a,b)=>{let x=a[k],y=b[k];if(k=="name")return sortF.asc?(x<y?-1:1):(x>y?-1:1);return sortF.asc?x-y:y-x;});
 document.querySelectorAll("#factors th").forEach(th=>{th.classList.toggle("s",th.dataset.k==k);th.classList.toggle("asc",th.dataset.k==k&&sortF.asc);});
 const tb=$("factors").querySelector("tbody");tb.innerHTML=rows.map(r=>`<tr><td>${r.name}</td><td class="${r.sharpe>=0?'good':'bad'}">${f2(r.sharpe)}</td><td class="${r.ann>=0?'good':'bad'}">${(r.ann>=0?'+':'')+r.ann.toFixed(1)}</td></tr>`).join("");}
function renderPicks(){
 if(S=="wave"){
  $("curHdr").textContent="Current picks — as of "+PK.asof;
  let rows=PK.wave_now.map(x=>`<tr><td>${x.t}</td><td>${x.score}</td><td>${x.mom6>=0?'+':''}${x.mom6}%</td><td>$${x.px}</td></tr>`).join("");
  $("curBody").innerHTML=`<table><thead><tr><th>Ticker</th><th>ML %ile</th><th>6-mo mom</th><th>Price</th></tr></thead><tbody>${rows}</tbody></table>`;
  $("curNote").textContent="Model output as of the latest data month — fundamentals lag, not investment advice. ~12 equal-weight names.";
  $("histCard").classList.remove("hide");
 }else{
  $("curHdr").textContent="Current book — as of "+PK.asof;
  const tag=a=>a.map(t=>`<span class="pill">${t}</span>`).join("");
  $("curBody").innerHTML=`<div style="font-size:12px;color:var(--mut);font-weight:700;margin:0 0 6px">LONG (top decile)</div>${tag(PK.summit_long)}
   <div style="font-size:12px;color:var(--mut);font-weight:700;margin:12px 0 6px">SHORT (bottom decile)</div>${tag(PK.summit_short)}`;
  $("curNote").textContent="Representative top/bottom names of a broad ~decile market-neutral book, as of the latest data month. Not advice.";
  $("histCard").classList.add("hide");
 }}
function renderHist(){let rows=PK.wave_hist.slice();const k=sortH.k;rows.sort((a,b)=>{let x=a[k],y=b[k];if(k=="t"||k=="entry")return sortH.asc?(x<y?-1:1):(x>y?-1:1);return sortH.asc?x-y:y-x;});
 document.querySelectorAll("#hist th").forEach(th=>{th.classList.toggle("s",th.dataset.k==k);th.classList.toggle("asc",th.dataset.k==k&&sortH.asc);});
 $("hist").querySelector("tbody").innerHTML=rows.map(r=>`<tr><td>${r.t}</td><td>${r.entry}</td><td>${r.held}</td><td class="good">+${r.ret}%</td></tr>`).join("");}
function render(){const c=CFG[S];$("badge").textContent=c.badge;$("title").textContent=c.name;$("subtitle").textContent=c.sub;$("lgS").textContent=c.name;$("info").innerHTML=c.info;
 $("tab-wave").classList.toggle("on",S=="wave");$("tab-summit").classList.toggle("on",S=="summit");
 renderMetrics();chart();renderPicks();renderHist();renderAnnual();renderFactors();}
function setStrat(s){S=s;render();}
function setP(i){P=i;document.querySelectorAll("#chips button").forEach((b,j)=>b.classList.toggle("on",j==i));renderMetrics();chart();}
$("chips").innerHTML=PERIODS.map((p,i)=>`<button class="${i==0?'on':''}" onclick="setP(${i})">${p[0]}</button>`).join("");
document.querySelectorAll("#annual th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortA.k==k)sortA.asc=!sortA.asc;else{sortA.k=k;sortA.asc=k=="year";}renderAnnual();});
document.querySelectorAll("#factors th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortF.k==k)sortF.asc=!sortF.asc;else{sortF.k=k;sortF.asc=false;}renderFactors();});
document.querySelectorAll("#hist th").forEach(th=>th.onclick=()=>{const k=th.dataset.k;if(sortH.k==k)sortH.asc=!sortH.asc;else{sortH.k=k;sortH.asc=(k=="t"||k=="entry");}renderHist();});
render();
</script></body></html>"""
html=html.replace("__DATA__",DATA).replace("__PICKS__",PICKS)
open("/home/user/bonds/docs/wave-summit.html","w").write(html)
# --- independent SUMMIT-only page ---
solo=(html
  .replace("<title>WAVE & SUMMIT — Strategy Brief</title>","<title>SUMMIT — Market-Neutral Alpha</title>")
  .replace('<header><h1>WAVE &amp; SUMMIT</h1><p>Equity strategy brief · survivorship-clean point-in-time · interactive</p></header>',
           '<header><h1>SUMMIT</h1><p>Market-neutral long/short · pure alpha · survivorship-clean point-in-time</p></header>')
  .replace('<nav class="wrap"><button id="tab-wave" class="on" onclick="setStrat(\'wave\')">WAVE</button><button id="tab-summit" onclick="setStrat(\'summit\')">SUMMIT</button></nav>',
           '<nav class="wrap" style="display:none"><button id="tab-wave"></button><button id="tab-summit"></button></nav>')
  .replace('let S="wave",P=0','let S="summit",P=0')
  .replace('<footer>WAVE = deploy now (long-only). SUMMIT = bigger alpha, when you can short.<br>Survivorship-clean &amp; point-in-time; null-gauntlet validated. Research, not advice.</footer>',
           '<footer>SUMMIT — market-neutral long/short, ~0 correlation to the market.<br>Survivorship-clean &amp; point-in-time; all-in net of modeled costs. Research, not advice.</footer>'))
open("/home/user/bonds/docs/summit.html","w").write(solo)
open("/home/user/bonds/dca/research/strategies/strategies.html","w").write(html)
print("written",len(html),"bytes")
