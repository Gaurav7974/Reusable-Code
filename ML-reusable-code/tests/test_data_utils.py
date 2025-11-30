import pytest
from data.cleaning import clean_text

def test_clean_text_basic():
    raw = "Hello!!! Visit http://example.com"
    out = clean_text(raw)
    assert "http" not in out
    assert "Hello" in out
