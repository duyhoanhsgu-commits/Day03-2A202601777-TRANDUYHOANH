"""
Tools module for E-commerce ReAct Agent.
Defines stock checking, discount validation, and shipping cost calculation tools.
"""

from typing import Dict, Any, Union

# Sample Inventory Database
STOCK_DATABASE: Dict[str, Dict[str, Any]] = {
    "iphone": {"price": 20000000, "stock": 15, "status": "in_stock"},
    "iphone 15": {"price": 20000000, "stock": 15, "status": "in_stock"},
    "iphone 15 pro": {"price": 28000000, "stock": 8, "status": "in_stock"},
    "macbook": {"price": 35000000, "stock": 5, "status": "in_stock"},
    "macbook pro": {"price": 45000000, "stock": 3, "status": "in_stock"},
    "airpods": {"price": 4000000, "stock": 25, "status": "in_stock"},
    "samsung galaxy": {"price": 18000000, "stock": 0, "status": "out_of_stock"},
}

# Sample Discount Coupon Database
COUPON_DATABASE: Dict[str, int] = {
    "WINNER": 10,       # 10% discount
    "WELCOME10": 10,   # 10% discount
    "TECH50": 15,      # 15% discount
    "SUPERDEAL": 20,   # 20% discount
}

# Sample Shipping Costs Database / Rule Matrix
SHIPPING_RATES: Dict[str, Dict[str, Any]] = {
    "hà nội": {"base_cost": 30000, "estimated_days": 2},
    "hanoi": {"base_cost": 30000, "estimated_days": 2},
    "tp.hcm": {"base_cost": 20000, "estimated_days": 1},
    "ho chi minh": {"base_cost": 20000, "estimated_days": 1},
    "đà nẵng": {"base_cost": 25000, "estimated_days": 2},
    "danang": {"base_cost": 25000, "estimated_days": 2},
}


def check_stock(item_name: str) -> Dict[str, Any]:
    """
    Check price, stock quantity, and availability status of an item.

    Args:
        item_name (str): Name of the item/product.

    Returns:
        dict: {"price": float/int, "stock": int, "status": str}
    """
    key = item_name.strip().lower()
    
    # Try exact match or partial match
    if key in STOCK_DATABASE:
        data = STOCK_DATABASE[key]
        return {
            "item_name": item_name,
            "price": data["price"],
            "stock": data["stock"],
            "status": data["status"]
        }

    # Partial matching for common item names
    for stock_key, data in STOCK_DATABASE.items():
        if stock_key in key or key in stock_key:
            return {
                "item_name": stock_key,
                "price": data["price"],
                "stock": data["stock"],
                "status": data["status"]
            }

    return {
        "item_name": item_name,
        "price": 0,
        "stock": 0,
        "status": "not_found"
    }


def get_discount(coupon_code: str) -> Dict[str, Any]:
    """
    Validate a coupon code and get its discount percentage.

    Args:
        coupon_code (str): Promotional coupon code string.

    Returns:
        dict: {"discount_percent": int, "valid": bool}
    """
    code_clean = coupon_code.strip().upper()
    if code_clean in COUPON_DATABASE:
        discount = COUPON_DATABASE[code_clean]
        return {
            "coupon_code": code_clean,
            "discount_percent": discount,
            "valid": True
        }
    
    return {
        "coupon_code": coupon_code,
        "discount_percent": 0,
        "valid": False
    }


def calc_shipping(weight: Union[float, int], destination: str) -> Dict[str, Any]:
    """
    Calculate shipping cost and estimated delivery time based on weight and destination.

    Args:
        weight (float): Weight in kg.
        destination (str): Destination city/region.

    Returns:
        dict: {"shipping_cost": float/int, "estimated_days": int}
    """
    dest_clean = destination.strip().lower()
    
    # Base rate lookup
    rate_info = None
    for dest_key, info in SHIPPING_RATES.items():
        if dest_key in dest_clean or dest_clean in dest_key:
            rate_info = info
            break
            
    if not rate_info:
        # Default standard shipping rate if destination unknown
        rate_info = {"base_cost": 50000, "estimated_days": 3}

    try:
        weight_num = float(weight)
    except (ValueError, TypeError):
        weight_num = 1.0

    # Weight surcharge calculation: base cost + 5,000 VND per kg after 1kg
    extra_weight = max(0.0, weight_num - 1.0)
    shipping_cost = int(rate_info["base_cost"] + extra_weight * 5000)

    return {
        "destination": destination,
        "weight_kg": weight_num,
        "shipping_cost": shipping_cost,
        "estimated_days": rate_info["estimated_days"]
    }


# Map of available tools for registry / ReAct agent
AVAILABLE_TOOLS = {
    "check_stock": check_stock,
    "get_discount": get_discount,
    "calc_shipping": calc_shipping,
}

TOOL_SCHEMAS = [
    {
        "name": "check_stock",
        "description": "Tra cứu giá cả (price), số lượng tồn kho (stock) và trạng thái hàng (status) của sản phẩm. Argument: item_name (string)",
        "func": check_stock,
    },
    {
        "name": "get_discount",
        "description": "Kiểm tra mã giảm giá (coupon_code) và trả về phần trăm giảm giá (discount_percent) cùng tính hợp lệ (valid). Argument: coupon_code (string)",
        "func": get_discount,
    },
    {
        "name": "calc_shipping",
        "description": "Tính phí vận chuyển (shipping_cost) và số ngày dự kiến giao (estimated_days) dựa trên khối lượng (weight) và địa điểm (destination). Argument: weight (number/float), destination (string)",
        "func": calc_shipping,
    },
]
