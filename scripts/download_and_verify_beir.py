"""Verified acquisition of all three official BEIR datasets. Never disables TLS."""

from scripts.full_beir_reevaluation import verify

if __name__ == "__main__":
    verify()
