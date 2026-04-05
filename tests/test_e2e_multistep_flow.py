import pytest

from app import ChatRequest, ask


class DummyRequest:
    def __init__(self, headers=None):
        self.headers = headers or {}
        self.client = None


class FakeRouter:
    async def process_request(self, message, user_id=None, session_id=None, context=None, intent=None):
        canonical = (intent or "chat").lower()
        if canonical == "api_call":
            canonical = "api"

        metadata = {"flow": canonical}
        if canonical == "cart":
            metadata["model_used"] = "api"
        if canonical == "search":
            metadata["grounding"] = {
                "grounded": True,
                "evidence_count": 1,
                "evidence_ids": ["prd_001"],
                "citations": [{"product_id": "prd_001", "name": "OnePlus 12"}],
            }

        return {
            "response": f"ok:{canonical}",
            "intent": canonical,
            "confidence": 0.9,
            "metadata": metadata,
            "session_id": session_id,
        }


@pytest.mark.asyncio
async def test_e2e_multistep_transaction_flow():
    router = FakeRouter()
    user_id = "user_001"
    session_id = "session_001"

    steps = [
        ("search", "tim dien thoai samsung"),
        ("cart", "them vao gio hang"),
        ("checkout", "toi muon checkout"),
        ("payment", "thanh toan don #12345"),
        ("shipping", "tracking don #12345"),
        ("refund", "hoan tien don #12345"),
        ("return", "doi tra don #12345"),
    ]

    for idx, (intent, message) in enumerate(steps, start=1):
        headers = {"idempotency-key": f"step-{idx}"} if intent != "search" else {}
        request = ChatRequest(
            message=message,
            user_id=user_id,
            session_id=session_id,
            intent=intent,
        )
        response = await ask(request, DummyRequest(headers=headers), router)

        assert response.intent == intent
        assert response.response.startswith("ok:")
        assert response.metadata is not None
        assert response.metadata.get("flow") == intent

    # Verify idempotency replay for a transaction step
    request = ChatRequest(
        message="toi muon checkout",
        user_id=user_id,
        session_id=session_id,
        intent="checkout",
    )
    first = await ask(request, DummyRequest(headers={"idempotency-key": "checkout-replay"}), router)
    replay = await ask(request, DummyRequest(headers={"idempotency-key": "checkout-replay"}), router)

    assert first.intent == "checkout"
    assert replay.metadata is not None
    assert replay.metadata.get("idempotency", {}).get("replayed") is True
