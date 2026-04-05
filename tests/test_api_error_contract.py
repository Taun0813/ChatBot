import pytest

from core.api_model import APIModel


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404, 500])
async def test_order_error_contract(status_code: int):
    model = APIModel({"enable_api_calls": True})

    async def fake_call(**kwargs):
        return {"error": f"Service error: {status_code}", "status_code": status_code}

    model._call_spring_boot_service = fake_call  # type: ignore[method-assign]

    response = await model.handle_order_request("Kiểm tra đơn #12345")

    if status_code == 401:
        assert "đăng nhập" in response.lower()
    else:
        assert "xin lỗi" in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404, 500])
async def test_payment_error_contract(status_code: int):
    model = APIModel({"enable_api_calls": True})

    async def fake_call(**kwargs):
        return {"error": f"Service error: {status_code}", "status_code": status_code}

    model._call_spring_boot_service = fake_call  # type: ignore[method-assign]

    response = await model.handle_payment_request("Thanh toán đơn #12345")

    if status_code == 401:
        assert "đăng nhập" in response.lower()
    else:
        assert "xin lỗi" in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404, 500])
async def test_warranty_error_contract(status_code: int):
    model = APIModel({"enable_api_calls": True})

    async def fake_call(**kwargs):
        return {"error": f"Service error: {status_code}", "status_code": status_code}

    model._call_spring_boot_service = fake_call  # type: ignore[method-assign]

    response = await model.handle_warranty_request("Bảo hành sản phẩm 6789")

    if status_code == 401:
        assert "đăng nhập" in response.lower()
    else:
        assert "xin lỗi" in response.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404, 500])
async def test_cart_error_contract(status_code: int):
    model = APIModel({"enable_api_calls": True})

    async def fake_call(**kwargs):
        return {"error": f"Service error: {status_code}", "status_code": status_code}

    model._call_spring_boot_service = fake_call  # type: ignore[method-assign]

    response = await model.handle_cart_request("thêm vào giỏ product id: prd_001", context={"jwt_token": "x"})

    if status_code == 401:
        assert "đăng nhập" in response.lower()
    elif status_code == 403:
        assert "không có quyền" in response.lower()
    elif status_code == 404:
        assert "không tìm thấy" in response.lower()
    else:
        assert "xin lỗi" in response.lower()
