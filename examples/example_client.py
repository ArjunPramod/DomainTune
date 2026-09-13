"""Minimal example client for the DomainTune API.

Usage:
    uv run python examples/example_client.py
"""
import httpx

API_URL = "http://localhost:8000"

SAMPLE_OCR_TEXT = """\
TAN WOON YANN
BOOK TA .K(TAMAN DAYA) SDN BND
789417-W
NO.53 55,57 & 59, JALAN SAGU 18,
TAMAN DAYA, 81100 JOHOR BAHRU,
JOHOR.
DOCUMENT NO : TD01167104
DATE: 25/12/2018 8:13:39 PM
CASHIER: MANIS
MEMBER: CASH BILL
TOTAL: 9.00"""


def main():
    with httpx.Client(timeout=60.0) as client:
        health = client.get(f"{API_URL}/health")
        print("Health:", health.json())

        response = client.post(f"{API_URL}/extract", json={"ocr_text": SAMPLE_OCR_TEXT})
        response.raise_for_status()
        print("Extraction result:")
        print(response.json())


if __name__ == "__main__":
    main()
