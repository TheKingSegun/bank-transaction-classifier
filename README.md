# Nigerian Bank Transaction Classifier

An NLP + rule-based ML pipeline that automatically parses and categorises Nigerian bank statement transactions — the core engine powering **VenaNow**, an AI personal finance platform for the Nigerian market.

## Project Overview

Nigerian bank statements (Zenith, GTB, Access, First Bank, UBA) follow inconsistent narration formats. This project builds a robust parser and multi-class classifier to categorise transactions into meaningful spending categories.

## Transaction Categories
- `AIRTIME_DATA` — MTN, Airtel, Glo, 9mobile top-ups
- `FOOD_DINING` — Restaurants, food delivery (Glovo, Chowdeck)
- `TRANSPORT` — Bolt, Uber, fuel, BRT
- `TRANSFERS` — NIP/NEFT to individuals vs. businesses
- `UTILITY_BILLS` — NEPA/DISCO, water, DSTV, internet
- `SALARY_INCOME` — Regular credit patterns
- `POS_PURCHASE` — Point-of-sale retail transactions
- `ATM_WITHDRAWAL` — Cash withdrawals
- `BANK_CHARGES` — COT, SMS, card fees

## Performance
| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| Rule-based baseline | 61.2% | 0.58 |
| TF-IDF + Logistic Regression | 82.4% | 0.80 |
| Fine-tuned classifier | 89.1% | 0.87 |

## Tools & Technologies
- Python: scikit-learn, spaCy, regex, pandas
- PDF parsing: pdfplumber, PyMuPDF
- Model serving: FastAPI

## How to Run
```bash
pip install -r requirements.txt
python src/parser.py --input data/sample_statement.pdf
```
