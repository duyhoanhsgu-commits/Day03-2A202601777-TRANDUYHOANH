import os
import sys
import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.tools import check_stock, get_discount, calc_shipping


def test_check_stock_existing_item():
    """Test check_stock for an existing product."""
    result = check_stock("iPhone")
    assert result["price"] == 20000000
    assert result["stock"] == 15
    assert result["status"] == "in_stock"


def test_check_stock_non_existing_item():
    """Test check_stock for a non-existing product."""
    result = check_stock("Unknown Product 999")
    assert result["price"] == 0
    assert result["stock"] == 0
    assert result["status"] == "not_found"


def test_get_discount_valid_code():
    """Test get_discount with a valid promotional code."""
    result = get_discount("WINNER")
    assert result["discount_percent"] == 10
    assert result["valid"] is True


def test_get_discount_invalid_code():
    """Test get_discount with an invalid promotional code."""
    result = get_discount("INVALID_CODE_123")
    assert result["discount_percent"] == 0
    assert result["valid"] is False


def test_calc_shipping_hanoi():
    """Test calc_shipping for Hanoi destination."""
    result = calc_shipping(1.0, "Hà Nội")
    assert result["shipping_cost"] == 30000
    assert result["estimated_days"] == 2


def test_calc_shipping_heavy_weight():
    """Test calc_shipping for item heavier than 1kg."""
    result = calc_shipping(3.0, "Hà Nội")
    # Base: 30000, extra weight: 2kg * 5000 = 10000 -> total: 40000
    assert result["shipping_cost"] == 40000
    assert result["estimated_days"] == 2


if __name__ == "__main__":
    pytest.main(["-v", __file__])
