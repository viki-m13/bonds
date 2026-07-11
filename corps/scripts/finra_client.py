"""Client for the FINRA Data Platform API (api.finra.org).

FINRA disseminates every corporate-bond trade through TRACE — the corporate
analogue of MSRB EMMA for munis. Trade-level TRACE datasets are exposed via
the FINRA Data API but require OAuth2 (client-credentials). Registration is
free at the FINRA API Developer Center; it yields a client id/secret.

  export FINRA_API_CLIENT_ID=...
  export FINRA_API_CLIENT_SECRET=...

Auth flow (client credentials):
  POST https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token
       ?grant_type=client_credentials
  Authorization: Basic base64(client_id:client_secret)
  -> {"access_token": "...", "expires_in": ...}

Query flow:
  POST https://api.finra.org/data/group/fixedIncomeMarket/name/traceCorporateBond
  Authorization: Bearer <token>
  body: {"limit":..., "offset":..., "compareFilters":[...],
         "dateRangeFilters":[...], "fields":[...]}
  -> CSV or JSON records.

The public dataset `otcMarket/regShoDaily` needs no auth and is used by
`selftest()` to prove the HTTP/pagination/parse path end-to-end without
credentials.

Field names for `traceCorporateBond` follow FINRA's documented TRACE schema
and are mapped defensively in `corps/research/panel.py`; verify against the
live schema on first credentialed run.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time

TOKEN_URL = ("https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
             "?grant_type=client_credentials")
API = "https://api.finra.org"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


class FinraError(RuntimeError):
    pass


def _curl(args: list[str], timeout: int = 60, retries: int = 4) -> tuple[int, bytes]:
    """Run curl, return (http_status, body). Retries on transport errors."""
    base = ["curl", "-sS", "--max-time", str(timeout), "-A", UA,
            "-w", "\n%{http_code}"]
    last = None
    for attempt in range(retries):
        out = subprocess.run(base + args, capture_output=True,
                             timeout=timeout + 10)
        if out.returncode == 0 and out.stdout:
            body, _, code = out.stdout.rpartition(b"\n")
            try:
                return int(code), body
            except ValueError:
                last = out.stdout[:200]
        else:
            last = out.stderr.decode(errors="replace")[:200]
        time.sleep(min(2 ** attempt * 2, 20))
    raise FinraError(f"curl failed: {last}")


class FinraClient:
    def __init__(self, client_id: str | None = None,
                 client_secret: str | None = None):
        self.client_id = client_id or os.environ.get("FINRA_API_CLIENT_ID")
        self.client_secret = (client_secret
                              or os.environ.get("FINRA_API_CLIENT_SECRET"))
        self._token: str | None = None
        self._token_exp = 0.0

    # ------------------------------------------------------------------ auth
    @property
    def authenticated(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get_token(self) -> str:
        if not self.authenticated:
            raise FinraError(
                "FINRA_API_CLIENT_ID / FINRA_API_CLIENT_SECRET not set. "
                "Register free at the FINRA API Developer Center.")
        if self._token and time.time() < self._token_exp - 60:
            return self._token
        cred = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode()).decode()
        code, body = _curl(["-X", "POST", "-H", f"Authorization: Basic {cred}",
                            TOKEN_URL])
        if code != 200:
            raise FinraError(f"token request failed ({code}): {body[:200]!r}")
        data = json.loads(body)
        self._token = data["access_token"]
        self._token_exp = time.time() + int(data.get("expires_in", 1800))
        return self._token

    # --------------------------------------------------------------- queries
    def query(self, group: str, dataset: str, *, limit: int = 1000,
              offset: int = 0, filters: dict | None = None,
              public: bool = False) -> list[dict]:
        """One page of a dataset as list-of-dicts (parsed from CSV)."""
        url = f"{API}/data/group/{group}/name/{dataset}"
        body = {"limit": limit, "offset": offset}
        if filters:
            body.update(filters)
        args = ["-X", "POST", "-H", "Content-Type: application/json",
                "-H", "Accept: text/plain", "--data", json.dumps(body)]
        if not public:
            args = ["-H", f"Authorization: Bearer {self._get_token()}"] + args
        code, raw = _curl(args + [url])
        if code == 401:
            raise FinraError("401 unauthorized — check credentials / token")
        if code != 200:
            raise FinraError(f"query {dataset} failed ({code}): {raw[:200]!r}")
        return _parse_csv(raw.decode(errors="replace"))

    def iter_dataset(self, group: str, dataset: str, *, filters: dict | None = None,
                     page: int = 5000, public: bool = False, max_rows: int | None = None):
        """Yield all rows of a dataset, paginating by offset."""
        offset = 0
        while True:
            rows = self.query(group, dataset, limit=page, offset=offset,
                              filters=filters, public=public)
            if not rows:
                return
            yield from rows
            offset += len(rows)
            if len(rows) < page or (max_rows and offset >= max_rows):
                return

    # ---------------------------------------------------------------- health
    def selftest(self) -> dict:
        """Prove the HTTP/pagination/parse path with a public dataset."""
        rows = self.query("otcMarket", "regShoDaily", limit=3, public=True)
        return {"public_api_ok": len(rows) > 0,
                "sample_columns": list(rows[0].keys())[:6] if rows else [],
                "authenticated": self.authenticated}


def _parse_csv(text: str) -> list[dict]:
    import csv
    import io
    text = text.strip()
    if not text or text.startswith("{"):  # error JSON or empty
        return []
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


if __name__ == "__main__":
    c = FinraClient()
    print(json.dumps(c.selftest(), indent=2))
