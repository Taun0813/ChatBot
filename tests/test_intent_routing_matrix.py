import pytest

from core.router import AgnoRouter, RouterConfig


@pytest.fixture
def router() -> AgnoRouter:
    config = RouterConfig(
        rag_config={},
        interaction_config={},
        api_config={},
        personalization_config={},
        hybrid_config={"enable_hybrid": False},
    )
    return AgnoRouter(config)


@pytest.mark.parametrize(
    "message,expected_intent",
    [
        ("Tôi muốn tìm điện thoại Samsung dưới 20 triệu", "search"),
        ("Đơn hàng #12345 của tôi tới đâu rồi", "order"),
        ("Theo dõi shipping đơn #12345", "shipping"),
        ("Thanh toán đơn #12345 bằng momo", "payment"),
        ("Kiểm tra bảo hành sản phẩm 9988", "warranty"),
        ("Thêm vào giỏ hàng iPhone 15", "cart"),
        ("Tôi muốn checkout đơn này", "checkout"),
        ("Tôi muốn hoàn tiền đơn #12345", "refund"),
        ("Tôi muốn đổi trả đơn #12345", "return"),
        ("Xin chào bạn", "chat"),
    ],
)
def test_rule_router_intent_matrix(router: AgnoRouter, message: str, expected_intent: str):
    detected = router._route_request(message)
    assert detected == expected_intent


@pytest.mark.parametrize(
    "raw_intent,canonical",
    [
        ("api_call", "api"),
        ("product_search", "search"),
        ("order_inquiry", "order"),
        ("payment_question", "payment"),
        ("warranty_inquiry", "warranty"),
        ("shipping_inquiry", "shipping"),
        ("cart_management", "cart"),
        ("checkout_request", "checkout"),
        ("refund_request", "refund"),
        ("return_request", "return"),
    ],
)
def test_intent_alias_normalization(router: AgnoRouter, raw_intent: str, canonical: str):
    assert router._normalize_intent(raw_intent) == canonical
