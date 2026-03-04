# Pre-Disclosure Price Discovery in Korean Earnings Announcements

Evidence of Information Leakage Under Fair Disclosure Regulation

## Abstract

This study investigates pre-disclosure price discovery in Korean earnings announcements using 2,221 fair disclosure filings from 64 KOSPI-listed companies (2020–2026). Despite Korea's Fair Disclosure regulation, 91% of cumulative abnormal returns accumulate *before* the official disclosure date. Abnormal trading volume emerges 3 days prior to disclosure.

## Key Findings

- **CAR[-5,-1] = +0.64%** (t=5.67, p<0.001) — 91% of total event window CAR
- **CAR[0,+1] = +0.01%** (p=0.94) — negligible event-day reaction
- **Abnormal volume** begins Day -3 (1.05x, p=0.004), peaks Day +1 (1.71x)
- **Structural shift**: positive drift 2020–2022 → negative drift 2024–2026
- **Quarterly pattern**: Q1 negative (-0.51%), Q2 positive (+0.60%)

## Data

- **Source**: DART (dart.fss.or.kr) via OpenDartReader API
- **Sample**: 2,221 earnings disclosures, 64 companies, 12 chaebol groups
- **Period**: January 2020 – March 2026
- **Stock prices**: FinanceDataReader (KOSPI index + individual stocks)

## Repository Structure

```
code/           # Analysis scripts
paper/          # LaTeX manuscript + figures
data/           # Data files (not tracked, see Data section)
```

## Author

Yongjun Kim — Ajou University, College of Software Convergence
