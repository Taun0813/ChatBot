# Walkthrough - Order Service Debug

## The Issue
Users encountered an error when querying order status (e.g., "Where is order 1234?").
**Error Log**: `Failed to get order info: 'str' object has no attribute 'get'`

## Investigation
- **Cause 1**: `core/api_model.py` was iterating over dictionary **keys** instead of the `orders` list in the mock data.
- **Cause 2**: Mock data keys were in **camelCase** (matching Spring Boot service) but the Python code expected **snake_case** keys during lookup.
- **Cause 3**: Resulting mock object was raw camelCase, causing subsequent formatting functions to fail (expecting snake_case).

## The Fix
Modified `core/api_model.py` (`_get_order_info`, `_get_payment_info`, `_get_warranty_info`) to:
1. Iterate over the correct list (e.g., `mock_orders.get("orders", [])`).
2. Search using both `id` (snake_case) and `orderId` (camelCase) keys.
3. Apply `_transform_*_response` methods to convert raw mock data (camelCase) into the expected internal format (snake_case) before returning.

## Verification
Created a new test script `tests/test_mock_services.py` that verifies:
- Order lookup via Mock Service.
- Payment lookup via Mock Service.
- Warranty lookup via Mock Service.

### Verification Output
```
--- Testing Order Info ---
Order Result: True
Order ID: 1234

--- Testing Payment Info ---
Payment Result: True

--- Testing Warranty Info ---
Warranty Result: True
```

## Files Changed
- `core/api_model.py`: Apply logic fixes.
- `docs/PLAN-debug-order-service.md`: Implementation plan.
- `tests/test_mock_services.py`: New test script.
