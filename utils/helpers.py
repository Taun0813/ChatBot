"""
Helper utilities for the AI Agent system
"""

import re
from typing import Any, Optional

def format_price(price: float, currency: str = "VND") -> str:
    """Format price with currency"""
    if currency == "VND":
        return f"{price:,.0f} {currency}"
    else:
        return f"{price:.2f} {currency}"

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]', '', text)
    
    return text

def validate_input(text: str, max_length: int = 1000) -> bool:
    """Validate user input"""
    if not text or len(text.strip()) == 0:
        return False
    
    if len(text) > max_length:
        return False
    
    return True