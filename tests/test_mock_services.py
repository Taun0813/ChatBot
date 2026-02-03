import sys
import os
import asyncio
import logging

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.api_model import APIModel

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    model = APIModel()
    
    print("--- Testing Order Info ---")
    try:
        result = await model._get_order_info("1234")
        print(f"Order Result: {result is not None}")
        if result:
            print(f"Order ID: {result.get('order_id')}")
    except Exception as e:
        print(f"Order Error: {e}")

    print("\n--- Testing Payment Info ---")
    try:
        # Note: method renamed to _get_payment_info
        result = await model._get_payment_info("1234")
        print(f"Payment Result: {result is not None}")
    except Exception as e:
        print(f"Payment Error: {e}")

    print("\n--- Testing Warranty Info ---")
    try:
        # Note: method renamed to _get_warranty_info
        # Use a product id from mock data or just check non-crash
        result = await model._get_warranty_info("iphone_15_128gb") 
        print(f"Warranty Result: {result is not None}")
    except Exception as e:
        print(f"Warranty Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
