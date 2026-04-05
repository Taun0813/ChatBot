"""
Product Schema Definition - E-commerce

This module defines the canonical product contract used by dataset processing,
RAG ingestion, and product services.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import re

DEFAULT_CURRENCY = "VND"
USD_TO_VND_RATE = 25_000

# Canonical categories used across the system.
PRODUCT_CATEGORIES = [
    "Điện thoại",
    "Laptop",
    "Tablet",
    "Phụ kiện",
    "Đồng hồ thông minh",
    "Tai nghe",
    "Sạc dự phòng",
    "Khác",
]

_CATEGORY_ALIASES = {
    "phone": "Điện thoại",
    "mobile": "Điện thoại",
    "smartphone": "Điện thoại",
    "smartphones": "Điện thoại",
    "điện thoại": "Điện thoại",
    "dien thoai": "Điện thoại",
    "laptop": "Laptop",
    "laptops": "Laptop",
    "notebook": "Laptop",
    "macbook": "Laptop",
    "tablet": "Tablet",
    "ipad": "Tablet",
    "phụ kiện": "Phụ kiện",
    "phu kien": "Phụ kiện",
    "accessory": "Phụ kiện",
    "accessories": "Phụ kiện",
    "watch": "Đồng hồ thông minh",
    "smartwatch": "Đồng hồ thông minh",
    "headphone": "Tai nghe",
    "headphones": "Tai nghe",
    "earphone": "Tai nghe",
    "tai nghe": "Tai nghe",
    "audio": "Tai nghe",
    "power bank": "Sạc dự phòng",
    "sac du phong": "Sạc dự phòng",
    "sạc dự phòng": "Sạc dự phòng",
}

_SPEC_KEY_ALIASES = {
    "screen_size": "màn hình",
    "display": "màn hình",
    "resolution": "độ phân giải",
    "refresh_rate": "tần số quét",
    "processor": "chip",
    "processor_brand": "chip",
    "chip": "chip",
    "ram": "ram",
    "memory": "ram",
    "storage": "rom",
    "internal_memory": "rom",
    "rom": "rom",
    "battery": "pin",
    "battery_capacity": "pin",
    "capacity": "pin",
    "camera": "camera",
    "rear_camera": "camera sau",
    "rear_cameras": "camera sau",
    "front_camera": "camera trước",
    "front_cameras": "camera trước",
    "weight": "trọng lượng",
    "os": "hệ điều hành",
    "operating_system": "hệ điều hành",
    "5g": "5g",
    "nfc": "nfc",
    "fast_charging": "sạc nhanh",
    "fast charging": "sạc nhanh",
    "ir_blaster": "ir blaster",
}


def normalize_text(value: Any, default: str = "") -> str:
    """Normalize any scalar value to a trimmed string."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_brand(brand: Any) -> str:
    """Normalize brand text for display."""
    text = normalize_text(brand, default="Khác")
    return text.title() if text.islower() else text


def normalize_category(category: Any) -> str:
    """Normalize category to the canonical ecommerce category set."""
    text = normalize_text(category, default="Khác").lower()
    if text in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[text]

    compact_text = re.sub(r"\s+", " ", text).strip()
    if compact_text in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[compact_text]

    return category if category in PRODUCT_CATEGORIES else "Khác"


def parse_numeric_value(value: Any) -> float:
    """Extract a numeric value from mixed text such as '6.1 inches' or '3,600mAh'."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else 0.0


def _detect_currency(value: Any, default: str = DEFAULT_CURRENCY) -> str:
    """Detect currency from a price-like value."""
    if isinstance(value, str):
        text = value.strip().upper()
        if "USD" in text or "$" in text:
            return "USD"
        if "VND" in text or "VNĐ" in text:
            return "VND"
        if "INR" in text:
            return "INR"
        if "PKR" in text:
            return "PKR"
        if "AED" in text:
            return "AED"
        if "CNY" in text:
            return "CNY"
    return default


def parse_price_value(value: Any, currency_hint: Optional[str] = None) -> Tuple[float, str, int]:
    """Parse a price value and normalize it to VND.

    Returns: (source_price, source_currency, price_vnd)
    """
    if value is None:
        return 0.0, DEFAULT_CURRENCY, 0

    source_currency = (currency_hint or _detect_currency(value)).upper()
    source_price = parse_numeric_value(value)

    if source_price <= 0:
        return 0.0, source_currency, 0

    if source_currency == "USD":
        return source_price, source_currency, int(source_price * USD_TO_VND_RATE)
    if source_currency == "VND":
        return source_price, source_currency, int(source_price)

    # Fallback: preserve the numeric value for currencies we do not explicitly convert.
    return source_price, source_currency, int(source_price)


def normalize_availability(value: Any) -> str:
    """Normalize availability text to a small controlled vocabulary."""
    text = normalize_text(value, default="In Stock").lower()
    if any(token in text for token in ["in stock", "còn hàng", "available", "available now"]):
        return "In Stock"
    if any(token in text for token in ["out of stock", "hết hàng", "sold out"]):
        return "Out of Stock"
    if any(token in text for token in ["preorder", "pre-order", "đặt trước"]):
        return "Preorder"
    return normalize_text(value, default="In Stock")


def infer_stock(availability: Any, explicit_stock: Optional[Any] = None) -> int:
    """Infer stock from explicit quantity or availability string."""
    if explicit_stock is not None:
        try:
            return max(0, int(float(explicit_stock)))
        except (TypeError, ValueError):
            pass

    normalized = normalize_availability(availability).lower()
    if normalized == "in stock":
        return 10
    if normalized == "preorder":
        return 1
    return 0


def normalize_specifications(specifications: Any) -> Dict[str, Any]:
    """Normalize specifications into a flat, searchable dictionary."""
    if specifications is None:
        return {}

    if isinstance(specifications, str):
        parsed: Dict[str, Any] = {}
        for chunk in re.split(r"[;\n]", specifications):
            if ":" not in chunk:
                continue
            raw_key, raw_value = chunk.split(":", 1)
            key = normalize_text(raw_key).lower()
            mapped_key = _SPEC_KEY_ALIASES.get(key, key)
            parsed[mapped_key] = normalize_text(raw_value)
        return parsed

    if not isinstance(specifications, dict):
        return {}

    flattened: Dict[str, Any] = {}

    generic_groups = {
        "display",
        "performance",
        "camera",
        "battery",
        "connectivity",
        "general",
        "specs",
        "features",
    }

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                child_name = normalize_text(child_key).lower()
                mapped_key = _SPEC_KEY_ALIASES.get(child_name, child_name)
                next_prefix = mapped_key if prefix in generic_groups or prefix == mapped_key else (
                    f"{prefix} {mapped_key}".strip() if prefix else mapped_key
                )
                _walk(next_prefix, child_value)
            return

        if isinstance(value, list):
            flattened[prefix] = ", ".join(normalize_text(item) for item in value if normalize_text(item))
            return

        flattened[prefix] = normalize_text(value)

    for key, value in specifications.items():
        normalized_key = normalize_text(key).lower()
        mapped_key = _SPEC_KEY_ALIASES.get(normalized_key, normalized_key)
        _walk(mapped_key, value)

    return {k: v for k, v in flattened.items() if k}


def derive_features_from_specifications(specifications: Dict[str, Any], price_vnd: int = 0) -> List[str]:
    """Create coarse-grained features from normalized specifications."""
    features: List[str] = []

    ram = parse_numeric_value(specifications.get("ram"))
    battery = parse_numeric_value(specifications.get("pin"))
    camera = parse_numeric_value(specifications.get("camera"))

    if camera >= 50:
        features.append("camera cao cấp")
    elif camera >= 20:
        features.append("camera tốt")

    if battery >= 5000:
        features.append("pin khỏe")
    elif battery >= 4000:
        features.append("pin tốt")

    if ram >= 8:
        features.append("ram cao")
        features.append("đa nhiệm tốt")

    if price_vnd >= 20_000_000:
        features.append("cao cấp")
        features.append("sang trọng")
    elif price_vnd and price_vnd < 5_000_000:
        features.append("giá rẻ")

    lower_keys = {normalize_text(key).lower() for key in specifications.keys()}
    if "5g" in lower_keys:
        features.append("5G")
    if "sạc nhanh" in lower_keys:
        features.append("sạc nhanh")

    return list(dict.fromkeys(features))


def build_product_id(brand: str, name: str, fallback_prefix: str = "product") -> str:
    """Build a stable product identifier."""
    base = f"{normalize_text(brand).lower()}_{normalize_text(name).lower()}"
    base = re.sub(r"[^a-z0-9_-]+", "_", base)
    base = re.sub(r"_+", "_", base).strip("_")
    return base or f"{fallback_prefix}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"


@dataclass
class ProductSchema:
    """Canonical product contract for ecommerce data."""

    id: str
    name: str
    description: str
    category: str
    price: float
    brand: str
    features: List[str] = field(default_factory=list)
    specifications: Dict[str, Any] = field(default_factory=dict)
    images: List[str] = field(default_factory=list)
    currency: str = DEFAULT_CURRENCY
    price_vnd: int = 0
    source_price: float = 0.0
    source_currency: str = DEFAULT_CURRENCY
    availability: str = "In Stock"
    stock: int = 0
    rating: float = 0.0
    reviews_count: int = 0
    tags: List[str] = field(default_factory=list)
    source: Optional[str] = None
    source_id: Optional[str] = None
    backend_id: Optional[str] = None
    is_live: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        payload = asdict(self)
        if self.created_at:
            payload["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            payload["updated_at"] = self.updated_at.isoformat()
        return payload

    @classmethod
    def from_raw(
        cls,
        raw: Dict[str, Any],
        *,
        default_category: str = "Khác",
        source: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> "ProductSchema":
        """Create a normalized product from raw dataset input."""
        raw = raw or {}

        name = normalize_text(
            raw.get("name")
            or raw.get("model")
            or raw.get("Model Name")
            or raw.get("title"),
            default="Unknown Product",
        )
        brand = normalize_brand(
            raw.get("brand")
            or raw.get("company")
            or raw.get("brand_name")
            or raw.get("Company Name")
        )

        category = normalize_category(
            raw.get("category") or raw.get("type") or default_category
        )

        price_source = (
            raw.get("price_vnd")
            or raw.get("price")
            or raw.get("Launched Price (USA)")
            or raw.get("Launched Price (India)")
            or raw.get("Launched Price (China)")
            or raw.get("Launched Price (Dubai)")
            or raw.get("Launched Price (Pakistan)")
        )
        source_currency = _detect_currency(price_source)
        source_price, detected_currency, price_vnd = parse_price_value(
            price_source,
            currency_hint=source_currency,
        )
        currency = DEFAULT_CURRENCY if price_vnd else detected_currency

        raw_specifications = raw.get("specifications") or raw.get("specs") or {}
        specifications = normalize_specifications(raw_specifications)

        if not specifications:
            fallback_spec_fields = {
                "màn hình": raw.get("Screen Size") or raw.get("screen_size"),
                "ram": raw.get("RAM") or raw.get("ram_capacity"),
                "rom": raw.get("internal_memory") or raw.get("ROM"),
                "pin": raw.get("Battery Capacity") or raw.get("battery_capacity"),
                "camera trước": raw.get("Front Camera") or raw.get("primary_camera_front"),
                "camera": raw.get("Back Camera") or raw.get("primary_camera_rear"),
                "chip": raw.get("Processor") or raw.get("processor_brand"),
                "trọng lượng": raw.get("Mobile Weight") or raw.get("weight"),
                "hệ điều hành": raw.get("os"),
            }
            specifications = {
                key: normalize_text(value)
                for key, value in fallback_spec_fields.items()
                if normalize_text(value)
            }

        image = raw.get("image_url") or raw.get("image") or raw.get("images") or []
        if isinstance(image, str):
            images = [image] if image else []
        elif isinstance(image, list):
            images = [normalize_text(item) for item in image if normalize_text(item)]
        else:
            images = []

        availability = normalize_availability(raw.get("availability") or raw.get("stock_status"))
        stock = infer_stock(availability, raw.get("stock") or raw.get("inventory"))
        rating = float(raw.get("rating") or 0.0)
        reviews_count = int(raw.get("reviews_count") or raw.get("review_count") or 0)
        features = raw.get("features") or []
        if isinstance(features, str):
            features = [item.strip() for item in features.split(",") if item.strip()]
        features = list(dict.fromkeys([normalize_text(item) for item in features if normalize_text(item)]))

        if not features:
            features = derive_features_from_specifications(specifications, price_vnd=price_vnd)

        product_id = normalize_text(
            raw.get("id") or raw.get("backend_id") or raw.get("product_id") or build_product_id(brand, name),
            default=build_product_id(brand, name),
        )

        description = normalize_text(raw.get("description") or raw.get("desc"))
        if not description:
            price_text = f"{price_vnd:,} VNĐ" if price_vnd else normalize_text(price_source)
            description = f"{name} - {brand} - {category}. Giá {price_text}."

        created_at = raw.get("created_at")
        updated_at = raw.get("updated_at")

        parsed_created_at = None
        if isinstance(created_at, str) and created_at:
            try:
                parsed_created_at = datetime.fromisoformat(created_at)
            except ValueError:
                parsed_created_at = None

        parsed_updated_at = None
        if isinstance(updated_at, str) and updated_at:
            try:
                parsed_updated_at = datetime.fromisoformat(updated_at)
            except ValueError:
                parsed_updated_at = None

        tags = raw.get("tags") or []
        if isinstance(tags, str):
            tags = [item.strip() for item in tags.split(",") if item.strip()]

        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            id=product_id,
            name=name,
            description=description,
            category=category,
            price=float(price_vnd or source_price),
            brand=brand,
            features=features,
            specifications=specifications,
            images=images,
            currency=currency,
            price_vnd=int(price_vnd or source_price),
            source_price=float(source_price),
            source_currency=detected_currency,
            availability=availability,
            stock=stock,
            rating=rating,
            reviews_count=reviews_count,
            tags=[normalize_text(item) for item in tags if normalize_text(item)],
            source=source,
            source_id=source_id or normalize_text(raw.get("source_id") or raw.get("backend_id")),
            backend_id=normalize_text(raw.get("backend_id") or raw.get("id"), default=product_id),
            is_live=bool(raw.get("is_live", True)),
            created_at=parsed_created_at,
            updated_at=parsed_updated_at,
            metadata=metadata,
        )
