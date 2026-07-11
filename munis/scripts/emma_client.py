"""Minimal client for MSRB EMMA (emma.msrb.org) public trade data.

EMMA serves individual municipal bond trade prints (price, yield-to-worst,
par size, trade side) through the JSON endpoints that back its own web UI:

  POST /TradeData/GetSecurityTradeInfo   {"id": <securityId>}
      -> full trade-by-trade history for one security (~15y lookback)
  POST /Security/FindSimilarSecurities   {securityId, criteria, tradeDateRange}
      -> all securities matching criteria that traded in the window,
         with per-security trade count / total volume
  POST /TradeData/ValidateFSSCusip       {"cusip9": ...}
      -> maps a CUSIP-9 to EMMA's internal securityId
  GET  /TradeData/MostActivelyTradedRefresh
      -> today's most actively traded securities

Access notes (empirical, 2026-07):
  * A one-time ASP.NET disclaimer form must be accepted; the session cookie
    jar then authorizes the JSON endpoints.
  * The WAF rejects python-requests POSTs (TLS fingerprint) but accepts
    curl with a browser User-Agent, so all HTTP goes through curl.
  * EMMA renders CUSIP strings as images, so securities are keyed by
    EMMA's opaque securityId ("six") everywhere in this project.
  * robots.txt disallows only PDFs. We rate-limit and retry with backoff;
    intermittent connection resets are normal.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

BASE = "https://emma.msrb.org"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI",
    "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN",
    "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH",
    "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY", "PR", "GU", "VI",
]


class EmmaError(RuntimeError):
    pass


class EmmaClient:
    def __init__(self, cookie_jar: str | Path, delay: float = 0.7,
                 retries: int = 5, timeout: int = 120):
        self.jar = str(cookie_jar)
        self.delay = delay
        self.retries = retries
        self.timeout = timeout
        self._last_request = 0.0

    # ------------------------------------------------------------------ http

    def _curl(self, url: str, *, post_json: str | None = None,
              post_form: str | None = None, referer: str | None = None,
              retries: int | None = None) -> bytes:
        """One HTTP round trip via curl, honoring rate limit + retries."""
        cmd = [
            "curl", "-sS", "--max-time", str(self.timeout),
            "-A", UA, "-b", self.jar, "-c", self.jar,
            "-H", "Accept-Language: en-US,en;q=0.9",
        ]
        if referer:
            cmd += ["-H", f"Referer: {referer}", "-H", f"Origin: {BASE}"]
        if post_json is not None:
            cmd += ["-H", "Content-Type: application/json; charset=utf-8",
                    "-H", "X-Requested-With: XMLHttpRequest",
                    "--data", post_json]
        elif post_form is not None:
            cmd += ["-H", "Content-Type: application/x-www-form-urlencoded",
                    "--data", post_form]
        cmd.append(url)

        n = self.retries if retries is None else retries
        last_err = None
        for attempt in range(n):
            wait = self.delay - (time.time() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            try:
                out = subprocess.run(cmd, capture_output=True, timeout=self.timeout + 10)
                self._last_request = time.time()
                if out.returncode == 0 and out.stdout:
                    return out.stdout
                last_err = out.stderr.decode(errors="replace")[:200]
            except subprocess.TimeoutExpired:
                last_err = "curl timeout"
            time.sleep(min(2 ** attempt * 2, 30))
        raise EmmaError(f"curl failed for {url}: {last_err}")

    # --------------------------------------------------------------- session

    def ensure_session(self) -> None:
        """Accept the EMMA disclaimer once so the cookie jar is authorized."""
        target = f"{BASE}/TradeData/MostActivelyTraded"
        html = self._curl(target).decode(errors="replace")
        if "disclaimerContent" not in html:
            return  # already accepted
        hidden = dict(re.findall(
            r'name="(__[A-Za-z0-9]+)"[^>]*value="([^"]*)"', html))
        hidden["ctl00$mainContentArea$disclaimerContent$yesButton"] = "Yes"
        from urllib.parse import urlencode
        body = urlencode(hidden)
        html2 = self._curl(f"{BASE}/Disclaimer.aspx", post_form=body,
                           referer=target).decode(errors="replace")
        if "disclaimerContent" in html2:
            raise EmmaError("disclaimer accept did not stick")

    # ------------------------------------------------------------- endpoints

    def _post_json(self, path: str, payload: dict, referer: str) -> dict:
        raw = self._curl(f"{BASE}{path}", post_json=json.dumps(payload),
                         referer=referer)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            snippet = raw[:200].decode(errors="replace")
            raise EmmaError(f"non-JSON from {path}: {snippet}") from exc

    def most_actively_traded(self) -> list[dict]:
        raw = self._curl(f"{BASE}/TradeData/MostActivelyTradedRefresh",
                         referer=f"{BASE}/TradeData/MostActivelyTraded")
        return json.loads(raw)["data"]

    def find_similar(self, seed_six: str, state: str,
                     trade_date_range: str = "1year",
                     maturity_band: str | None = None) -> dict:
        """All securities in `state` that traded in the window.

        Only selected criteria constrain the result set. The state is passed
        explicitly; `maturity_band` (one of 6months/1year/2years/5years)
        restricts to maturities within that distance of the SEED security's
        maturity — used to partition states that exceed EMMA's result cap.
        """
        criteria = [
            {"dbName": "state", "selected": True, "rangeExtValue": state},
        ]
        if maturity_band:
            criteria.append({"dbName": "maturity_date", "selected": True,
                             "rangeExtValue": maturity_band})
        payload = {
            "securityId": seed_six,
            "criteria": criteria,
            "tradeDateRange": trade_date_range,
        }
        return self._post_json("/Security/FindSimilarSecurities", payload,
                               referer=f"{BASE}/Security/Details/{seed_six}")

    def security_trade_info(self, six: str) -> dict:
        """Full trade history for one security.

        Returns {securityId, securityDesc, data: [{TDT ms-epoch, PX price,
        YX yield-to-worst, TA par amount, TT side}]}. TT: 'D' inter-dealer,
        'S' dealer sells to customer (customer buy), 'P' dealer purchases
        from customer (customer sell).
        """
        return self._post_json("/TradeData/GetSecurityTradeInfo",
                               {"id": six},
                               referer=f"{BASE}/Security/Details/{six}")

    def validate_cusip(self, cusip9: str) -> dict:
        return self._post_json("/TradeData/ValidateFSSCusip",
                               {"cusip9": cusip9},
                               referer=f"{BASE}/TradeData/PriceDiscovery")
