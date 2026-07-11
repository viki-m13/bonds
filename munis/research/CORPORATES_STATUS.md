# Corporate bonds via FINRA TRACE — exploration status

**Goal:** obtain individual corporate-bond trade prints (the TRACE tape, the
corporate analogue of EMMA) and port the KEYSTONE dislocation-reversion
strategy to corporates.

**Status: partially cracked, blocked on an agreement-cookie + bot-management
gate in this environment.** The strategy is data-source-agnostic and ports the
moment a TRACE feed is available.

## What corporate data looks like

Corporate bond trades are reported to FINRA and disseminated via **TRACE**.
Each print carries price, yield, size, and a side (customer-buy /
customer-sell / inter-dealer) — the *same structure* KEYSTONE relies on.
FINRA republishes TRACE through the Morningstar-powered **FINRA Bond Center**
(`finra-markets.morningstar.com`).

## What we cracked

- **Session mint** works: `GET /finralogin.jsp` returns the session cookies
  (`qs_wsid`, `SessionID`, `UsrID`, `UsrName`, `Instid`).
- **Search endpoint reachable**: `POST /bondSearch.jsp` returns `200` and
  echoes the search type (`{B:null}`), i.e. it accepts the request but our
  keyword payload does not yet match its expected schema.

## What blocks a clean scrape (in this sandbox)

1. **Client-side agreement cookie.** `BondCenter/Results.jsp` redirects to the
   market-data home until a disclaimer/agreement is accepted. Acceptance is
   handled by a minified webpack bundle that sets a cookie in the browser —
   not a simple server POST like EMMA's ASP.NET disclaimer.
2. **Cloudflare bot management.** Responses carry `__cf_bm` / `_cfuvid`;
   automated clients are fingerprinted.
3. **Browser automation is blocked here.** Driving headless Chromium through
   the environment's egress proxy resets the connection on the Morningstar
   host (`net::ERR_CONNECTION_RESET`), so we cannot capture the real XHR /
   accept the agreement the way we did for EMMA.

## Concrete paths forward (any one unblocks it)

- **FINRA Data API (recommended).** `api.finra.org` exposes TRACE datasets via
  free-registration OAuth2 credentials — a stable, documented feed, no scraping.
  Needs a client-id/secret provisioned once.
- **Real browser session.** Accept the Bond Center agreement in a normal
  browser, export the resulting cookie jar, and replay `bondSearch.jsp` +
  the per-CUSIP trade endpoint with it (the muni playbook, one gate deeper).
- **Licensed TRACE feed** (Bloomberg/ICE/vendor) for production.

## Why the strategy ports with zero research risk

KEYSTONE's engine (`munis/research/backtest.py`) operates on a generic panel of
`(security, date) → {customer-buy px, customer-sell px, inter-dealer px, ytw,
par, side}`. Corporate TRACE has identical semantics. Swapping the loader
(`emma_client.py` → a TRACE client) and re-pointing `panel.py` is the only
new code; the signal, the honesty harness, the matched control, and the IS/OOS
machinery are unchanged. Corporates additionally have **wider spreads and
richer credit dispersion** than munis, which typically *strengthens* a
dislocation-reversion signal — but that is a hypothesis to test on the data,
not a claim.
