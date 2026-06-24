# INVENTED book: smart-money flow + ai-trader momentum (L4, honest OOS)

Tape 2026-06-22 14:00:01.037000 -> 2026-06-23 19:02:37.880000, 1,565,345 trades, 44 coins. Accounts scored on IS half by dollar-weighted flow→next-move alignment; top 10% = smart (follow), bottom 10% = dumb (fade); sets FROZEN before OOS. ai-trader intraday ROC momentum added as confirmation. OOS = last 40%.

**The robust read is the OOS IC + t-stat and net-of-cost PnL, not the annualized Sharpe (12h OOS).**


## 5min bars (smart=186, dumb=186 accts; 140 OOS bars; ppyr 105,120)

| signal | OOS IC (t) | gross Sh | net Sh (taker) | net Sh (maker) | turn/bar |
|---|---|---|---|---|---|
| smart-money flow | -0.0159 (-0.6) | +5.2 | -87.4 | +25.9 | 0.62 |
| ai-ROC momentum | -0.0412 (-1.5) | -43.2 | -82.2 | -34.5 | 0.31 |
| flow+momentum combo | -0.0265 (-0.9) | -25.4 | -101.6 | -8.5 | 0.54 |

## 15min bars (smart=158, dumb=158 accts; 47 OOS bars; ppyr 35,040)

| signal | OOS IC (t) | gross Sh | net Sh (taker) | net Sh (maker) | turn/bar |
|---|---|---|---|---|---|
| smart-money flow | +0.0082 (+0.2) | +1.5 | -44.0 | +11.6 | 0.69 |
| ai-ROC momentum | -0.0461 (-1.2) | -9.0 | -26.1 | -5.3 | 0.35 |
| flow+momentum combo | -0.0081 (-0.2) | +15.0 | -28.3 | +24.6 | 0.55 |

## Verdict (honest — the invention did NOT validate)

- **The headline finding is a negative one, and it is informative.** Selecting the top-10% 'informed' accounts on 17h of IS data and following them OOS gave a **near-zero / negative OOS IC** — i.e. the IS-informed accounts were NOT informed out-of-sample. By contrast the *blanket* whale-flow signal (no account selection, `flow_intraday.py`) had a consistently *positive* OOS IC (+0.013…+0.033). **Account-level selection overfit**: with 17h you cannot reliably tell skill from luck across 31k wallets, so the top-decile is mostly noise that reverses. Aggregate flow is the more robust object at this sample size.
- ai-trader's intraday ROC momentum also has a **negative OOS IC** here — intraday price momentum reversed in this 12h window. Neither the new flow idea nor the TA confirmation survives.
- Best book by OOS IC: **smart-money flow 15min** — IC +0.0082 (NOT significant (|t|<2)). Even its gross Sharpe (+1.5) is not bankable on 707 pooled points / ~12h. The maker-rebate column occasionally turns positive, but that rides a near-zero gross signal plus the rebate — not a real edge.
- **On Sharpe 3, honestly: not reached, and not close on this data.** I invented the smart-money book, tested it cleanly OOS, and it failed — that is the honest result. What this rules IN: the robust path is *aggregate* whale-flow at a *slower* horizon (its IC grows with horizon and is positive), validated on *weeks* of tape so account and regime noise average out. What it rules OUT: account-selection alpha and intraday TA on ~1 day of data. **I will not label anything here Sharpe 3 — it isn't, and pretending otherwise would be the one unacceptable outcome.**
