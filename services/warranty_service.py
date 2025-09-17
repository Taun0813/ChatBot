"""
Warranty Service (Mock)
Handles warranty-related operations
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class WarrantyStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    CLAIMED = "claimed"
    VOID = "void"

@dataclass
class Warranty:
    id: str
    product_id: str
    customer_id: str
    serial_number: str
    purchase_date: datetime
    expiry_date: datetime
    status: WarrantyStatus
    terms: str
    created_at: datetime

class WarrantyService:
    """Mock service for managing warranties"""
    
    def __init__(self):
        self.warranties: Dict[str, Warranty] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample warranty data"""
        base_date = datetime.now()
        
        sample_warranties = [
            {
                "id": "warranty-001",
                "product_id": "laptop-001",
                "customer_id": "customer-001",
                "serial_number": "MBP2024001",
                "purchase_date": base_date - timedelta(days=30),
                "expiry_date": base_date + timedelta(days=335),  # 1 year warranty
                "status": WarrantyStatus.ACTIVE,
                "terms": "1 year manufacturer warranty covering hardware defects",
                "created_at": base_date - timedelta(days=30)
            },
            {
                "id": "warranty-002",
                "product_id": "phone-001",
                "customer_id": "customer-002",
                "serial_number": "IPH2024002",
                "purchase_date": base_date - timedelta(days=15),
                "expiry_date": base_date + timedelta(days=350),
                "status": WarrantyStatus.ACTIVE,
                "terms": "1 year manufacturer warranty covering hardware defects",
                "created_at": base_date - timedelta(days=15)
            }
        ]
        
        for warranty_data in sample_warranties:
            warranty = Warranty(**warranty_data)
            self.warranties[warranty.id] = warranty
    
    async def check_warranty(self, product_id: str, serial_number: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Check warranty information for a product"""
        try:
            # Find warranty by product_id and optionally serial_number
            for warranty in self.warranties.values():
                if warranty.product_id == product_id:
                    if serial_number is None or warranty.serial_number == serial_number:
                        return {
                            "id": warranty.id,
                            "product_id": warranty.product_id,
                            "serial_number": warranty.serial_number,
                            "status": warranty.status.value,
                            "purchase_date": warranty.purchase_date.isoformat(),
                            "expiry_date": warranty.expiry_date.isoformat(),
                            "terms": warranty.terms,
                            "is_active": warranty.status == WarrantyStatus.ACTIVE,
                            "days_remaining": (warranty.expiry_date - datetime.now()).days if warranty.status == WarrantyStatus.ACTIVE else 0
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking warranty: {e}")
            return None
    
    async def create_warranty(
        self,
        product_id: str,
        customer_id: str,
        serial_number: str,
        purchase_date: datetime,
        warranty_duration_days: int = 365
    ) -> Optional[Dict[str, Any]]:
        """Create a new warranty"""
        try:
            warranty_id = f"warranty-{uuid.uuid4().hex[:8]}"
            expiry_date = purchase_date + timedelta(days=warranty_duration_days)
            
            warranty = Warranty(
                id=warranty_id,
                product_id=product_id,
                customer_id=customer_id,
                serial_number=serial_number,
                purchase_date=purchase_date,
                expiry_date=expiry_date,
                status=WarrantyStatus.ACTIVE,
                terms=f"{warranty_duration_days} days manufacturer warranty",
                created_at=datetime.now()
            )
            
            self.warranties[warranty_id] = warranty
            
            logger.info(f"Created warranty {warranty_id} for product {product_id}")
            
            return {
                "id": warranty.id,
                "product_id": warranty.product_id,
                "serial_number": warranty.serial_number,
                "status": warranty.status.value,
                "purchase_date": warranty.purchase_date.isoformat(),
                "expiry_date": warranty.expiry_date.isoformat(),
                "terms": warranty.terms
            }
            
        except Exception as e:
            logger.error(f"Error creating warranty: {e}")
            return None
    
    async def get_customer_warranties(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all warranties for a customer"""
        try:
            customer_warranties = []
            
            for warranty in self.warranties.values():
                if warranty.customer_id == customer_id:
                    customer_warranties.append({
                        "id": warranty.id,
                        "product_id": warranty.product_id,
                        "serial_number": warranty.serial_number,
                        "status": warranty.status.value,
                        "purchase_date": warranty.purchase_date.isoformat(),
                        "expiry_date": warranty.expiry_date.isoformat(),
                        "is_active": warranty.status == WarrantyStatus.ACTIVE
                    })
            
            return customer_warranties
            
        except Exception as e:
            logger.error(f"Error getting customer warranties: {e}")
            return []
    
    async def claim_warranty(self, warranty_id: str, claim_reason: str) -> bool:
        """Claim warranty (mark as claimed)"""
        try:
            warranty = self.warranties.get(warranty_id)
            if not warranty:
                return False
            
            if warranty.status != WarrantyStatus.ACTIVE:
                logger.warning(f"Cannot claim warranty {warranty_id} with status {warranty.status.value}")
                return False
            
            warranty.status = WarrantyStatus.CLAIMED
            
            logger.info(f"Claimed warranty {warranty_id}: {claim_reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error claiming warranty: {e}")
            return False
    
    async def get_warranty_stats(self) -> Dict[str, Any]:
        """Get warranty statistics"""
        try:
            total_warranties = len(self.warranties)
            
            status_counts = {}
            active_count = 0
            expired_count = 0
            
            for warranty in self.warranties.values():
                status = warranty.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                if warranty.status == WarrantyStatus.ACTIVE:
                    active_count += 1
                elif warranty.expiry_date < datetime.now():
                    expired_count += 1
            
            return {
                "total_warranties": total_warranties,
                "active_warranties": active_count,
                "expired_warranties": expired_count,
                "status_distribution": status_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting warranty stats: {e}")
            return {}

# Global service instance
warranty_service = WarrantyService()
