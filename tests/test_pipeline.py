"""
Cambodia MSE Intelligence - Automated Test Suite
Unit and Integration Tests for Formulas, Alert Engines, and Data Models.

Run with standard library (no dependencies required):
    python3 -m unittest discover -s tests
Or with pytest (if installed):
    pytest
"""

import os
import sys
import math
import unittest
import duckdb
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "cambodia_mse.duckdb")

class TestCambodiaMSEPipeline(unittest.TestCase):

    def test_database_connectivity(self):
        """Verify DuckDB analytical warehouse connects and has core dimension tables."""
        self.assertTrue(os.path.exists(DB_PATH), f"Database not found at {DB_PATH}")
        con = duckdb.connect(DB_PATH, read_only=True)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        con.close()

        required_tables = [
            "dim_products",
            "dim_customers",
            "dim_dates_macro",
            "fct_sales_transactions",
            "fct_daily_revenue_margins",
            "fct_category_sales",
            "fct_inventory_turnover"
        ]
        for tbl in required_tables:
            self.assertIn(tbl, tables, f"Expected table '{tbl}' missing from DuckDB warehouse.")

    def test_dynamic_rop_formula(self):
        """Verify dynamic Reorder Point (ROP) mathematics."""
        lead_time = 4.0        # 4 days
        d_avg = 12.3           # 12.3 units/day
        z_score = 1.645        # 95% service level
        sigma_demand = 4.9     # standard deviation of daily sales

        # Safety Stock formula: Z * sigma_d * sqrt(L)
        safety_stock = z_score * sigma_demand * math.sqrt(lead_time)
        expected_rop = (lead_time * d_avg) + safety_stock

        self.assertGreater(safety_stock, 0, "Safety stock must be positive")
        self.assertGreater(expected_rop, (lead_time * d_avg), "ROP must strictly exceed expected lead time demand")
        self.assertAlmostEqual(expected_rop, 65.32, places=1)

    def test_eoq_cost_minimization(self):
        """Verify Economic Order Quantity (EOQ) formula."""
        annual_demand = 12.3 * 365  # ~4,489 units
        order_cost = 5.0            # $5 delivery fee
        unit_cost = 4.20            # $4.20/unit
        holding_rate = 0.18         # 18% annual holding cost

        holding_cost_per_unit = unit_cost * holding_rate
        eoq = math.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)

        self.assertGreater(eoq, 0, "EOQ must be positive")
        self.assertTrue(200 < eoq < 300, f"EOQ should be ~243 units, got {eoq}")

    def test_telegram_alert_formatter(self):
        """Verify Telegram Markdown push notification structure."""
        from alerts.telegram_worker import format_telegram_alert

        mock_row = {
            "sku_id": "SKU-BEV-001",
            "product_name": "Angkor Premium Beer 330ml Can",
            "product_name_khmer": "ស្រាបៀរអង្គរ កំប៉ុង ៣៣០មល",
            "category_id": "Beverages",
            "current_stock_on_hand": 15,
            "reorder_point_units": 39.0,
            "safety_stock_level": 12.0,
            "lead_time_days": 2,
            "recommended_reorder_qty": 63,
            "unit_cost_usd": 0.65,
            "unit_cost_khr": 2658.50
        }

        msg = format_telegram_alert(mock_row, location="Phnom Penh Central Branch")

        self.assertIn("[INVENTORY WARNING] Low Stock Alert", msg)
        self.assertIn("SKU-BEV-001", msg)
        self.assertIn("15 units", msg)
        self.assertIn("39.0 units", msg)
        self.assertIn("63 units", msg)
        self.assertIn("USD", msg)
        self.assertIn("៛", msg)

    def test_dual_currency_consistency(self):
        """Verify transaction records adhere to dual currency 100 Riel rounding conventions."""
        con = duckdb.connect(DB_PATH, read_only=True)
        res = con.execute("""
            SELECT 
                AVG(ABS(unit_selling_price_usd - (unit_selling_price_khr / applied_exchange_rate))) as avg_delta_usd,
                MAX(ABS(unit_selling_price_usd - (unit_selling_price_khr / applied_exchange_rate))) as max_delta_usd
            FROM fct_sales_transactions;
        """).fetchone()
        con.close()

        avg_delta, max_delta = res[0], res[1]
        self.assertLess(avg_delta, 0.02, f"Average FX discrepancy too large: ${avg_delta:.4f}")
        self.assertLess(max_delta, 0.05, f"Maximum FX discrepancy too large: ${max_delta:.4f}")

if __name__ == "__main__":
    unittest.main()
