"""
Payment Service
Handles payment-related operations
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class PaymentStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"

class PaymentMethod(Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    BANK_TRANSFER = "bank_transfer"
    CASH = "cash"

@dataclass
class Payment:
    id: str
    order_id: str
    customer_id: str
    amount: float
    payment_method: PaymentMethod
    status: PaymentStatus
    transaction_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    notes: Optional[str] = None

class PaymentService:
    """Service for managing payments"""
    
    def __init__(self):
        self.payments: Dict[str, Payment] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample payment data"""
        sample_payments = [
            {
                "id": "payment-001",
                "order_id": "order-001",
                "customer_id": "customer-001",
                "amount": 2899.98,
                "payment_method": PaymentMethod.CREDIT_CARD,
                "status": PaymentStatus.COMPLETED,
                "transaction_id": "txn_123456789",
                "created_at": datetime(2024, 1, 15, 10, 35),
                "updated_at": datetime(2024, 1, 15, 10, 40),
                "notes": "Payment processed successfully"
            },
            {
                "id": "payment-002",
                "order_id": "order-002",
                "customer_id": "customer-002",
                "amount": 1999.98,
                "payment_method": PaymentMethod.PAYPAL,
                "status": PaymentStatus.PROCESSING,
                "transaction_id": "txn_987654321",
                "created_at": datetime(2024, 1, 18, 15, 25),
                "updated_at": datetime(2024, 1, 18, 15, 30)
            }
        ]
        
        for payment_data in sample_payments:
            payment = Payment(**payment_data)
            self.payments[payment.id] = payment
    
    async def process_payment(
        self,
        order_id: str,
        customer_id: str,
        amount: float,
        payment_method: str,
        payment_details: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Process a payment"""
        try:
            payment_id = f"payment-{uuid.uuid4().hex[:8]}"
            
            # Validate payment method
            try:
                method = PaymentMethod(payment_method)
            except ValueError:
                logger.error(f"Invalid payment method: {payment_method}")
                return None
            
            # Simulate payment processing
            transaction_id = f"txn_{uuid.uuid4().hex[:12]}"
            
            # Create payment record
            payment = Payment(
                id=payment_id,
                order_id=order_id,
                customer_id=customer_id,
                amount=amount,
                payment_method=method,
                status=PaymentStatus.PROCESSING,
                transaction_id=transaction_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                notes=payment_details.get("notes") if payment_details else None
            )
            
            self.payments[payment_id] = payment
            
            # Simulate processing delay
            await self._simulate_payment_processing(payment)
            
            logger.info(f"Processed payment {payment_id} for order {order_id}")
            
            return {
                "id": payment.id,
                "order_id": payment.order_id,
                "amount": payment.amount,
                "payment_method": payment.payment_method.value,
                "status": payment.status.value,
                "transaction_id": payment.transaction_id,
                "created_at": payment.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing payment: {e}")
            return None
    
    async def _simulate_payment_processing(self, payment: Payment):
        """Simulate payment processing with external gateway"""
        import asyncio
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Simulate success/failure (90% success rate)
        import random
        if random.random() < 0.9:
            payment.status = PaymentStatus.COMPLETED
            payment.notes = "Payment processed successfully"
        else:
            payment.status = PaymentStatus.FAILED
            payment.notes = "Payment failed - insufficient funds"
        
        payment.updated_at = datetime.now()
    
    async def get_payment_status(self, payment_id: str) -> Optional[Dict[str, Any]]:
        """Get payment status"""
        try:
            payment = self.payments.get(payment_id)
            if not payment:
                return None
            
            return {
                "id": payment.id,
                "order_id": payment.order_id,
                "amount": payment.amount,
                "payment_method": payment.payment_method.value,
                "status": payment.status.value,
                "transaction_id": payment.transaction_id,
                "created_at": payment.created_at.isoformat(),
                "updated_at": payment.updated_at.isoformat(),
                "notes": payment.notes
            }
            
        except Exception as e:
            logger.error(f"Error getting payment status: {e}")
            return None
    
    async def get_payments_by_order(self, order_id: str) -> List[Dict[str, Any]]:
        """Get all payments for an order"""
        try:
            order_payments = []
            
            for payment in self.payments.values():
                if payment.order_id == order_id:
                    order_payments.append({
                        "id": payment.id,
                        "amount": payment.amount,
                        "payment_method": payment.payment_method.value,
                        "status": payment.status.value,
                        "transaction_id": payment.transaction_id,
                        "created_at": payment.created_at.isoformat()
                    })
            
            return order_payments
            
        except Exception as e:
            logger.error(f"Error getting payments by order: {e}")
            return []
    
    async def get_customer_payments(self, customer_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get payments for a customer"""
        try:
            customer_payments = []
            
            for payment in self.payments.values():
                if payment.customer_id == customer_id:
                    customer_payments.append({
                        "id": payment.id,
                        "order_id": payment.order_id,
                        "amount": payment.amount,
                        "payment_method": payment.payment_method.value,
                        "status": payment.status.value,
                        "created_at": payment.created_at.isoformat()
                    })
            
            # Sort by creation date (newest first)
            customer_payments.sort(key=lambda x: x["created_at"], reverse=True)
            
            return customer_payments[:limit]
            
        except Exception as e:
            logger.error(f"Error getting customer payments: {e}")
            return []
    
    async def refund_payment(self, payment_id: str, amount: Optional[float] = None, reason: Optional[str] = None) -> bool:
        """Process a refund"""
        try:
            payment = self.payments.get(payment_id)
            if not payment:
                return False
            
            # Only allow refunds for completed payments
            if payment.status != PaymentStatus.COMPLETED:
                logger.warning(f"Cannot refund payment {payment_id} with status {payment.status.value}")
                return False
            
            # Update payment status
            payment.status = PaymentStatus.REFUNDED
            payment.updated_at = datetime.now()
            
            if reason:
                payment.notes = f"Refunded: {reason}"
            
            logger.info(f"Refunded payment {payment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error refunding payment: {e}")
            return False
    
    async def get_payment_methods(self) -> List[Dict[str, Any]]:
        """Get available payment methods"""
        return [
            {
                "id": method.value,
                "name": method.value.replace("_", " ").title(),
                "enabled": True
            }
            for method in PaymentMethod
        ]
    
    async def get_payment_stats(self) -> Dict[str, Any]:
        """Get payment statistics"""
        try:
            total_payments = len(self.payments)
            
            status_counts = {}
            method_counts = {}
            total_amount = 0.0
            
            for payment in self.payments.values():
                status = payment.status.value
                method = payment.payment_method.value
                
                status_counts[status] = status_counts.get(status, 0) + 1
                method_counts[method] = method_counts.get(method, 0) + 1
                
                if payment.status == PaymentStatus.COMPLETED:
                    total_amount += payment.amount
            
            return {
                "total_payments": total_payments,
                "status_distribution": status_counts,
                "method_distribution": method_counts,
                "total_amount_processed": round(total_amount, 2),
                "success_rate": round(status_counts.get("completed", 0) / total_payments * 100, 2) if total_payments > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting payment stats: {e}")
            return {}

# Global service instance
payment_service = PaymentService()
