"""
Receipt & Invoice Line-Item Parser.

Parses raw OCR text lines into structured JSON data:
- Store name / header
- Date / timestamp
- Line items: description, quantity, unit price, total price, is_discount
- Financial summary: subtotal, tax, total
- Sanity validation: checks if sum(item_totals) matches parsed subtotal/total
"""

import re
from typing import Dict, List, Any, Optional, Tuple


class ReceiptParser:
    """Regex-powered parser to transform raw OCR text into structured receipt JSON."""

    # Matches prices like: 4.29, $4.29, -1.50, 14.99, 4,29 (European comma)
    PRICE_REGEX = re.compile(r"[-+]?\$?\s*(\d{1,4}[.,]\d{2})(?!\d)")

    # Matches quantities like "2 @ $3.50", "2x 4.99", "3 @ 1.99"
    QTY_AT_PRICE_REGEX = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:@|x|X)\s*\$?(\d{1,4}[.,]\d{2})"
    )

    # Keywords that signal metadata lines rather than purchased items
    SUMMARY_KEYWORDS = {
        "subtotal": ["subtotal", "sub total", "sub-total", "net amount", "taxable subtotal"],
        "tax": ["tax", "sales tax", "vat", "gst", "hst", "state tax", "tax amount"],
        "total": ["total", "balance due", "amount due", "grand total", "total due", "final total"],
        "payment": ["cash", "credit", "visa", "mastercard", "amex", "change", "change due", "debit"],
    }

    DATE_REGEX = re.compile(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
    )

    TIME_REGEX = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b")

    def __init__(self):
        pass

    def parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Parse raw OCR text into a structured receipt record.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        metadata = self._extract_metadata(lines)
        line_items, summary = self._extract_items_and_summary(lines)

        # Validation check: calculate difference between item sum and stated total
        calc_subtotal = round(sum(item["total_price"] for item in line_items), 2)
        total_diff = None
        if summary.get("total") is not None:
            expected = summary["subtotal"] if summary.get("subtotal") else summary["total"]
            total_diff = round(abs(calc_subtotal - expected), 2)

        return {
            "merchant_name": metadata.get("merchant_name"),
            "date": metadata.get("date"),
            "time": metadata.get("time"),
            "line_items": line_items,
            "financial_summary": {
                "parsed_subtotal": summary.get("subtotal"),
                "parsed_tax": summary.get("tax"),
                "parsed_total": summary.get("total"),
                "calculated_items_sum": calc_subtotal,
                "discrepancy": total_diff,
            },
            "raw_line_count": len(lines),
        }

    def _extract_metadata(self, lines: List[str]) -> Dict[str, Any]:
        """Extract merchant name, date, and timestamp from top lines."""
        merchant = None
        date_val = None
        time_val = None

        # Look in the top 5 lines for merchant name (skip pure numbers or short codes)
        for line in lines[:5]:
            clean = re.sub(r"[^A-Za-z0-9\s&'-]", "", line).strip()
            if len(clean) > 3 and not re.match(r"^\d+$", clean):
                # Don't pick lines with words like receipt, invoice, order
                if not any(w in clean.lower() for w in ["receipt", "invoice", "order", "welcome", "store #"]):
                    merchant = clean
                    break

        # Search for date and time in the full receipt text
        for line in lines:
            if not date_val:
                date_match = self.DATE_REGEX.search(line)
                if date_match:
                    date_val = date_match.group(1)
            if not time_val:
                time_match = self.TIME_REGEX.search(line)
                if time_match:
                    time_val = time_match.group(1)

        return {
            "merchant_name": merchant,
            "date": date_val,
            "time": time_val,
        }

    def _extract_items_and_summary(
        self, lines: List[str]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[float]]]:
        """Distinguish line items from financial summaries (subtotal/tax/total)."""
        line_items: List[Dict[str, Any]] = []
        summary: Dict[str, Optional[float]] = {
            "subtotal": None,
            "tax": None,
            "total": None,
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            lower = line.lower()

            # Skip decorative lines, empty lines, and greeting footers
            if re.match(r"^[-=_*~]{3,}$", line) or "thank you" in lower:
                i += 1
                continue

            # Check if this line is a summary row (Subtotal, Tax, Total, Payment)
            is_summary, summary_type, amount = self._check_summary_line(line)
            if is_summary:
                # If amount wasn't on the same line, check if next line is just the price
                if amount is None and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    next_prices = self._extract_all_prices(next_line)
                    if next_prices and len(next_line) < 15:
                        amount = next_prices[-1]
                        i += 1  # consume next line

                if summary_type and summary_type in summary and summary[summary_type] is None:
                    summary[summary_type] = amount
                i += 1
                continue

            # If line is payment method or card info
            if any(k in lower for k in self.SUMMARY_KEYWORDS["payment"]):
                i += 1
                continue

            # Check if this line is an isolated orphan price (e.g. Total price printed without label)
            prices_curr = self._extract_all_prices(line)
            if prices_curr and len(re.findall(r"[A-Za-z]", line)) == 0:
                # If total is not set yet and we've already parsed subtotal/tax, this is likely the total
                if summary["total"] is None and (summary["subtotal"] is not None or summary["tax"] is not None):
                    summary["total"] = prices_curr[-1]
                i += 1
                continue

            # If line is purely metadata (date/time, phone, address, header), skip without merging
            if self._is_metadata_line(line):
                i += 1
                continue

            # Strip leading date/time stamps if merged into item line
            date_match = self.DATE_REGEX.search(line)
            time_match = self.TIME_REGEX.search(line)
            if date_match and len(line) > 25:
                line = self.DATE_REGEX.sub("", line).strip()
            if time_match and len(line) > 25:
                line = self.TIME_REGEX.sub("", line).strip()
            line = re.sub(r"^(?:(?:DATE|TIME|STORE|TEL|PHONE)\s*[:#-]?\s*)+", "", line, flags=re.IGNORECASE).strip()

            # Check if current line contains a price
            parsed_item = self._parse_line_item(line)
            if parsed_item:
                line_items.append(parsed_item)
            elif i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Ensure next line is NOT a summary row, footer, or metadata before combining
                is_next_summary, _, _ = self._check_summary_line(next_line)
                if (
                    not is_next_summary
                    and not self._is_metadata_line(next_line)
                    and not any(k in next_line.lower() for k in self.SUMMARY_KEYWORDS["payment"])
                ):
                    combined = f"{line} {next_line}"
                    parsed_2line = self._parse_line_item(combined)
                    if parsed_2line:
                        line_items.append(parsed_2line)
                        i += 1  # Skip the next line as it was consumed

            i += 1

        return line_items, summary

    def _is_metadata_line(self, line: str) -> bool:
        """Check if line is purely metadata (store name, phone, address, date/time, receipt header)."""
        lower = line.lower()
        # Phone numbers
        if re.search(r"\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}", line):
            return True
        # Pure date/time lines (without prices)
        if ("date" in lower or "time" in lower) and (self.DATE_REGEX.search(line) or self.TIME_REGEX.search(line)):
            if not self._extract_all_prices(line):
                return True
        # Street addresses or web
        if any(term in lower for term in ["street", "st,", " ave", "road", "blvd", "hwy", "highway", "www.", ".com"]):
            if not self._extract_all_prices(line):
                return True
        # Receipt / transaction headers without prices
        if any(term in lower for term in ["invoice #", "receipt #", "order #", "store #", "cashier", "register #"]):
            if not self._extract_all_prices(line):
                return True
        return False

    def _check_summary_line(self, line: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """Identify if a line is Subtotal, Tax, Total, Payment, etc."""
        lower = line.lower()

        # Remove percentages like (8.25%) or 8.25% so they don't get treated as prices
        line_clean = re.sub(r"\(\s*\d+(?:\.\d+)?\s*%\s*\)|\d+(?:\.\d+)?\s*%", "", line)
        prices = self._extract_all_prices(line_clean)
        price_val = prices[-1] if prices else None

        # Check payment lines first
        for kw in self.SUMMARY_KEYWORDS["payment"]:
            if kw in lower:
                return True, "payment", price_val

        # Check subtotal / tax / total lines
        for stype in ["subtotal", "tax", "total"]:
            for kw in self.SUMMARY_KEYWORDS[stype]:
                if kw in lower:
                    # If line has both total and payment, total takes precedence
                    return True, stype, price_val

        # Ignore decorative receipt divider lines
        if re.match(r"^[-=_*~]{3,}$", line.strip()):
            return True, "divider", None

        return False, None, None

    def _parse_line_item(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a candidate item line into {name, quantity, unit_price, total_price}."""
        # Must have at least one price
        prices = self._extract_all_prices(line)
        if not prices:
            return None

        # Ignore lines that are pure numbers, barcodes, or dates
        if self.DATE_REGEX.search(line) or re.match(r"^[\d\s\-#*/]+$", line):
            return None

        total_price = prices[-1]
        quantity = 1.0
        unit_price = total_price

        # Check for qty @ unit price expression like: "2 @ 1.50"
        qty_match = self.QTY_AT_PRICE_REGEX.search(line)
        if qty_match:
            try:
                quantity = float(qty_match.group(1))
                unit_price = float(qty_match.group(2).replace(",", "."))
            except ValueError:
                pass
        elif len(prices) >= 2:
            # Often first price is unit price or qty, last is total
            unit_price = prices[0]

        # Extract item text description by stripping prices and trailing tax letters (e.g. 4.29 T)
        text_without_price = self.PRICE_REGEX.sub("", line)
        cleaned_name = re.sub(r"[@xX]\s*\d+[.,]\d{2}", "", text_without_price)
        cleaned_name = re.sub(r"\b[TtFfNn]\b$", "", cleaned_name)  # Tax flag letters
        cleaned_name = re.sub(r"[#*]+", "", cleaned_name).strip()

        # Item description should have at least 2 alphanumeric characters
        if len(re.findall(r"[A-Za-z0-9]", cleaned_name)) < 2:
            return None

        # Check if item is a discount/coupon
        is_discount = total_price < 0 or any(
            w in cleaned_name.lower() for w in ["discount", "coupon", "saving", "promo"]
        )

        return {
            "raw_line": line,
            "description": cleaned_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_price": total_price,
            "is_discount": is_discount,
        }

    def _extract_all_prices(self, text: str) -> List[float]:
        """Extract all price floats found in text."""
        matches = self.PRICE_REGEX.findall(text)
        results = []
        for m in matches:
            cleaned = m.replace(",", ".").replace("$", "").strip()
            try:
                results.append(float(cleaned))
            except ValueError:
                continue
        return results
