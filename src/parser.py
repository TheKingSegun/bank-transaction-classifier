"""
Nigerian Bank Statement Parser
Handles PDF extraction and narration normalisation for major Nigerian banks.
"""
import re
import pandas as pd
from typing import List, Dict, Optional

# Nigerian bank narration patterns
PATTERNS = {
    "AIRTIME_DATA": [
        r"(MTN|AIRTEL|GLO|9MOBILE|ETISALAT).*(AIRTIME|DATA|RECHARGE|VTU)",
        r"AIRTIME.*(PURCHASE|TOPUP|TOP.UP)",
        r"(RECHARGE|VTU|VTPASS).*(MTN|AIRTEL|GLO)",
    ],
    "FOOD_DINING": [
        r"(GLOVO|CHOWDECK|JUMIA FOOD|BOLT FOOD|UBER EATS)",
        r"(RESTAURANT|EATERY|KITCHEN|BUKKA|BUKA|CAFETERIA)",
        r"(CHICKEN REPUBLIC|TANTALIZERS|DOMINOS|KFC|SWEET SENSATION)",
    ],
    "TRANSPORT": [
        r"(BOLT|UBER|RIDA|INDRIVER|TAXIFY)",
        r"(FUEL|PMS|PETROL|DIESEL).*(STATION|FILLING)",
        r"(BRT|BUS RAPID|LAGBUS|LAGOS BUS)",
    ],
    "SALARY_INCOME": [
        r"(SALARY|PAYROLL|PAYSLIP|WAGES?).*(CREDIT|PAYMENT)",
        r"(MONTHLY PAY|STAFF SALARY|EMPLOYEE PAY)",
        r"(HR|PAYROLL).*(TRANSFER|CREDIT)",
    ],
    "UTILITY_BILLS": [
        r"(NEPA|PHCN|DISCO|IBEDC|EKEDC|AEDC|PHEDC).*(VEND|PAYMENT|RECHARGE)",
        r"(DSTV|GOTV|STARTIMES).*(SUBSCRIPTION|PAYMENT|RENEWAL)",
        r"(SPECTRANET|SWIFT|SMILE|MTN BROADBAND|AIRTEL BROADBAND)",
        r"(WATER BOARD|LAWMA|RCCG WATER)",
    ],
    "BANK_CHARGES": [
        r"(COT|COMMISSION ON TURNOVER)",
        r"(SMS ALERT|CARD MAINTENANCE|ANNUAL FEE|STAMP DUTY)",
        r"(ACCOUNT MAINTENANCE|SERVICE CHARGE|MANAGEMENT FEE)",
    ],
    "ATM_WITHDRAWAL": [
        r"(ATM|CASH WITHDRAWAL|WITHDRL)",
        r"(CASHOUT|CASH OUT)",
    ],
    "POS_PURCHASE": [
        r"(POS|POINT OF SALE|P\.O\.S)",
        r"(PURCHASE|PUR|WEB PURCHASE|ONLINE PURCHASE)",
    ],
    "TRANSFERS": [
        r"(NIP|NEFT|TRANSFER|TRF).*(FROM|TO)",
        r"(SENT TO|RECEIVED FROM|PAYMENT TO|CREDIT FROM)",
    ],
}

def normalise_narration(narration: str) -> str:
    """Uppercase, strip special chars, and standardise spacing."""
    text = narration.upper().strip()
    text = re.sub(r"[^A-Z0-9 /\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def rule_based_classify(narration: str) -> str:
    """Apply regex pattern matching for transaction classification."""
    normalised = normalise_narration(narration)
    for category, patterns in PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, normalised):
                return category
    return "MISCELLANEOUS"

def classify_transactions(df: pd.DataFrame, narration_col: str = "narration") -> pd.DataFrame:
    """Classify all transactions in a DataFrame."""
    df = df.copy()
    df["category"] = df[narration_col].apply(rule_based_classify)
    df["confidence"] = df["category"].apply(lambda x: 0.95 if x != "MISCELLANEOUS" else 0.40)
    return df

def parse_statement_text(raw_text: str) -> pd.DataFrame:
    """
    Parse raw text extracted from a Nigerian bank statement PDF.
    Handles GTB, Zenith, Access, First Bank, UBA formats.
    """
    rows = []
    # Generic pattern: date, narration, debit, credit, balance
    pattern = re.compile(
        r"(\d{2}[/-]\d{2}[/-]\d{4})"     # Date
        r"\s+(.+?)\s+"                      # Narration
        r"([\d,]+\.\d{2})?"               # Debit (optional)
        r"\s*([\d,]+\.\d{2})?"            # Credit (optional)
        r"\s+([\d,]+\.\d{2})"             # Balance
    )
    for match in pattern.finditer(raw_text):
        date, narration, debit, credit, balance = match.groups()
        rows.append({
            "date": date,
            "narration": narration.strip(),
            "debit": float(debit.replace(",", "")) if debit else 0.0,
            "credit": float(credit.replace(",", "")) if credit else 0.0,
            "balance": float(balance.replace(",", "")),
            "type": "DR" if debit else "CR",
        })
    return pd.DataFrame(rows)

def summarise_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Spending summary by category."""
    debits = df[df["debit"] > 0].copy()
    summary = debits.groupby("category").agg(
        total_spend=("debit", "sum"),
        transaction_count=("debit", "count"),
        avg_amount=("debit", "mean"),
    ).sort_values("total_spend", ascending=False)
    summary["spend_pct"] = (summary["total_spend"] / summary["total_spend"].sum() * 100).round(1)
    return summary

if __name__ == "__main__":
    # Demo with sample data
    sample = pd.DataFrame({
        "narration": [
            "NIP TRANSFER TO JOHN ADEYEMI GTB",
            "MTN AIRTIME VTU 08012345678",
            "BOLT TECHNOLOGY DEBIT",
            "GLOVO NIGERIA FOOD ORDER",
            "EKEDC ELECTRICITY VENDING",
            "ATM CASH WITHDRAWAL IKEJA",
            "POS PURCHASE SHOPRITE LEKKI",
            "COT COMMISSION ON TURNOVER",
            "SALARY CREDIT GLOVO NIGERIA",
            "SPECTRANET BROADBAND SUBSCRIPTION",
        ],
        "debit": [15000, 2000, 3500, 8200, 12000, 50000, 25000, 380, 0, 15000],
        "credit": [0, 0, 0, 0, 0, 0, 0, 0, 850000, 0],
    })
    result = classify_transactions(sample)
    print(result[["narration", "category", "confidence"]].to_string(index=False))
    print()
    print(summarise_by_category(result))
