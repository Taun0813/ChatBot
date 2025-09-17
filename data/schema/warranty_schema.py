"""
Warranty Schema Definition
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum

class WarrantyStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CLAIMED = "claimed"
    VOID = "void"

@dataclass
class WarrantySchema:
    id: str
    product_id: str
    customer_id: str
    serial_number: str
    purchase_date: datetime
    expiry_date: datetime
    status: WarrantyStatus
    terms: str
    created_at: datetime
