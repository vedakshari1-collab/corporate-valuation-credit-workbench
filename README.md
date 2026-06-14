# Corporate Valuation & Credit Risk Workbench

A professional Streamlit workbench for public-company financial statement analysis, DCF valuation, peer benchmarking, and transparent credit risk scoring.

This project is designed as a resume-grade corporate finance and full-stack Python project, not a quant trading system, stock prediction model, or algorithmic trading app.

## Author

Created and maintained by Vedakshari (`vedakshari1-collab`).

## Why This Project Matters

Finance teams, equity research analysts, credit analysts, and investment banking teams often need to move from raw filings to a clear investment view. This workbench demonstrates that workflow:

- Pull annual company facts from SEC EDGAR / XBRL where available.
- Fall back to stable sample data when external requests fail.
- Calculate core profitability, liquidity, leverage, cash flow, working capital, and DuPont ratios.
- Build a simple, editable DCF valuation.
- Compare public-company peers on consistent metrics.
- Generate an analyst-style Markdown investment memo.

## Project Structure

```text
corporate-valuation-credit-workbench/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── sample_financials.csv
│   └── sample_peers.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── sec_edgar.py
│   ├── financial_ratios.py
│   ├── valuation.py
│   ├── credit_score.py
│   ├── peer_analysis.py
│   ├── report_generator.py
│   └── utils.py
├── tests/
│   ├── test_ratios.py
│   ├── test_valuation.py
│   └── test_credit_score.py
└── outputs/
    ├── sample_report.md
    └── screenshots/
        └── README.md
```

## Features

### 1. Financial Statement Analysis

The app calculates:

- Revenue growth
- Gross margin, operating margin, and net profit margin
- ROA and ROE
- Debt-to-equity and debt-to-assets
- Current ratio, quick ratio, and cash ratio where available
- Free cash flow and FCF margin
- Interest coverage
- Asset turnover, equity multiplier, and DuPont-style ROE

### 2. Working Capital Analysis

The workbench estimates:

- Days Sales Outstanding
- Days Inventory Outstanding
- Days Payable Outstanding
- Cash Conversion Cycle
- Current assets versus current liabilities trend
- Operating cash flow versus net income

For financial institutions, some industrial working-capital metrics may be unavailable or less meaningful. The app handles those gaps without forcing misleading calculations.

### 3. Internal Educational Credit Score

The credit score is a transparent 0-100 framework across five equally weighted categories:

- Liquidity
- Leverage
- Coverage
- Profitability
- Cash flow quality

The output maps to:

- Strong
- Stable
- Watchlist
- Stressed

Important: this is not a rating-agency credit rating and should not be treated as investment advice.

### 4. DCF Valuation

The DCF module includes editable assumptions for:

- Base revenue
- Revenue growth
- Operating margin
- Tax rate
- FCF margin or reinvestment rate
- WACC
- Terminal growth
- Net debt
- Shares outstanding

Outputs include projected revenue, projected FCF, terminal value, enterprise value, equity value, fair value per share, upside/downside when price is available, and WACC / terminal growth sensitivity.

### 5. Peer Comparison

Enter 3-5 tickers such as:

```text
AAPL, MSFT, AMZN, JPM
```

The dashboard compares:

- Revenue growth
- Net margin
- ROE and ROA
- Debt-to-equity
- Current ratio
- FCF margin
- Internal credit score
- Estimated DCF fair value where possible

### 6. Investment Memo Export

The memo generator creates a Markdown report with:

- Company overview
- Financial performance
- Profitability analysis
- Liquidity and working capital analysis
- Leverage and credit view
- DCF valuation summary
- Peer comparison
- Key risks
- Analyst-style conclusion

## Data Sources

The preferred live data path is SEC EDGAR company facts:

- Company ticker to CIK mapping: `https://www.sec.gov/files/company_tickers.json`
- Company facts API: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`

The app also includes sample fallback data in `data/sample_financials.csv` for AAPL, MSFT, AMZN, and JPM. The sample data is included for demonstration and local reliability. It should not be treated as current market data or investment advice.

For responsible SEC API usage, you can set a user agent with contact information:

```powershell
$env:SEC_USER_AGENT="Your Name your.email@example.com"
```

## Setup

From the parent folder:

```powershell
cd corporate-valuation-credit-workbench
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Run Tests

```powershell
cd corporate-valuation-credit-workbench
python -m pytest
```

## Finance Methodology

### Free Cash Flow

```text
Free Cash Flow = Operating Cash Flow - Capital Expenditures
FCF Margin = Free Cash Flow / Revenue
```

### DuPont-Style ROE

```text
ROE ~= Net Margin x Asset Turnover x Equity Multiplier
```

This helps separate profitability, efficiency, and leverage.

### Cash Conversion Cycle

```text
DSO = Average Accounts Receivable / Revenue x 365
DIO = Average Inventory / Cost of Revenue x 365
DPO = Average Accounts Payable / Cost of Revenue x 365
CCC = DSO + DIO - DPO
```

### DCF

The DCF estimates enterprise value as:

```text
Enterprise Value = PV(Explicit Free Cash Flows) + PV(Terminal Value)
Equity Value = Enterprise Value - Net Debt
Fair Value Per Share = Equity Value / Shares Outstanding
```

Terminal value uses the Gordon Growth formula:

```text
Terminal Value = Final Year FCF x (1 + Terminal Growth) / (WACC - Terminal Growth)
```

## Screenshots

Screenshot placeholders are included in `outputs/screenshots/`. Add captured Streamlit screenshots there before publishing the project on GitHub.

Suggested screenshots:

- Home / Project Overview
- Single Company Analysis
- DCF Valuation
- Credit Risk Score
- Peer Comparison
- Investment Memo Export

## Resume Bullets

- Built a full-stack Python and Streamlit corporate finance workbench using SEC EDGAR/XBRL company facts, pandas analytics, Plotly visualizations, and sample-data fallback handling.
- Implemented financial statement analysis, working capital diagnostics, DuPont ROE decomposition, DCF valuation with sensitivity tables, peer benchmarking, and a transparent internal credit risk score.
- Developed modular, tested finance logic for ratio calculation, valuation, and credit scoring, plus automated Markdown investment memo generation for equity research and credit analysis workflows.

## Disclaimer

This project is for education, portfolio demonstration, and engineering practice. It does not provide investment advice, price predictions, trading signals, or official credit ratings.
