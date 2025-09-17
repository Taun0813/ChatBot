# services/__init__.py - Services Package
"""
Mock services for external integrations
"""

from .product_service import ProductService
from .order_service import OrderService
from .warranty_service import WarrantyService
from .payment_service import PaymentService

__all__ = [
    'ProductService',
    'OrderService',
    'WarrantyService',
    'PaymentService'
]
