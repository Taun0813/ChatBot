from core.router import AgnoRouter, RouterConfig


def _make_router() -> AgnoRouter:
    return AgnoRouter(
        RouterConfig(
            rag_config={},
            interaction_config={},
            api_config={},
            personalization_config={},
            hybrid_config={"enable_hybrid": False},
        )
    )


def test_grounding_metadata_has_valid_evidence_fields():
    router = _make_router()
    products = [
        {
            "backend_id": "prd_001",
            "name": "OnePlus 12 256GB",
            "brand": "OnePlus",
            "category": "Smartphone",
            "price_vnd": 23990000,
            "availability": "Còn hàng",
            "similarity_score": 0.91,
            "source": "products_export",
            "source_id": "12345",
        },
        {
            "id": "prd_002",
            "name": "Samsung S24",
            "brand": "Samsung",
            "category": "Smartphone",
            "price": 21990000,
            "availability": "Còn hàng",
            "similarity_score": 0.88,
        },
    ]

    grounding = router._build_grounding_metadata("tim dien thoai", products, max_citations=3)

    assert grounding["grounded"] is True
    assert grounding["evidence_count"] == 2
    assert len(grounding["citations"]) == 2

    for citation in grounding["citations"]:
        assert citation["product_id"]
        assert citation["name"]
        assert isinstance(citation["price_vnd"], int)
        assert citation["source"]


def test_grounding_metadata_empty_products_is_not_grounded():
    router = _make_router()
    grounding = router._build_grounding_metadata("tim dien thoai", [], max_citations=3)

    assert grounding["grounded"] is False
    assert grounding["evidence_count"] == 0
    assert grounding["citations"] == []
