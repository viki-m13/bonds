# Improvement work — shared harness note
Files (in scripts/): strategy.py (base VOLT: vol-targeted TQQQ|GLD-TLT, functions
tqqq_weight/strat_ret/dca/lump_stats, panel _etf_panel.pkl with 37 base + 16 reconstructed
leveraged ETFs incl TMF=3xTLT, UGL=2xGLD, SOXL, TECL...). dotcom.py (QQQ back to 1999,
reconstructs TQQQ, cash defense from FRED T-bill — use this to test 1999-2026 incl dot-com).
etf_panel.py (builds panel + validates recon vs real TQQQ 0.999).

## The bar (a change only counts if it clears ALL):
- Compare vs BOTH base VOLT and QQQ-DCA, ERA-SLICED: dot-com 2000-02, 2003-09, 2010-14,
  2015-19, 2020-26, full 1999-26 (or 2005-26 where bond/gold ETFs needed). DCA $1000/mo.
- Honest lump-sum $1 risk: CAGR, Sharpe, max drawdown (report the true worst case incl dot-com
  where testable), worst-12m.
- NO look-ahead: any vol/signal read at PRIOR month-end (shift(1)).
- Phase-robust: check rebalance-day sensitivity (the killer test).
- Improvement must hold ACROSS eras, not just one — flag if it's regime-specific/overfit.
- Base VOLT reference (2006-26): 2.41x QQQ-DCA, CAGR 22.7%, Sharpe 0.84, maxDD -47%
  (true -65% incl dot-com). Beat that on risk-adjusted return without wrecking any era.
