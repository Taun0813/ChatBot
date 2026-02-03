# PLAN-debug-order-service

## Goal
Fix `AttributeError: 'str' object has no attribute 'get'` when accessing order information and resolve "HTTP client not initialized" warnings.

## Analysis
- **Crash Cause**: `core/api_model.py` iterates over `mock_orders` (a dictionary) directly, effectively iterating over its keys. It should iterate over `mock_orders["orders"]`.
- **Warning Cause**: `app.py` disables API calls (`enable_api_calls: False`) but the application logs a warning when the client is not initialized during fallback attempts.

## Proposed Changes

### Core Logic
#### [MODIFY] [core/api_model.py](file:///d:/Code/NCKH/ChatBot/core/api_model.py)
- Update `_get_order_info` to safely access the "orders" list from the mock data dictionary.
- Update `_get_payment_info` and `_get_warranty_info` if they share the same pattern.
- Ensure `initialize()` is robust or checks are handled gracefully.

### Configuration (Optional)
#### [MODIFY] [app.py](file:///d:/Code/NCKH/ChatBot/app.py)
- Verify `enable_api_calls` setting and potentially comments on why it is disabled.

## Verification Plan

### Automated Reproduction
1. Create `tests/reproduce_issue.py` that imports `APIModel`, initializes it, and calls `_get_order_info("1234")`.
2. Run `python tests/reproduce_issue.py` -> Should fail before fix.
3. Run `python tests/reproduce_issue.py` -> Should pass after fix.

### Manual Verification
1. Start app: `python app.py`
2. Send Request:
   ```json
   POST /ask
   {
       "message": "đơn hàng 1234 ở đâu"
   }
   ```
3. Expect response with order details, not error.
