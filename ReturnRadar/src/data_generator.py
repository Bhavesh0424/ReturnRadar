"""Generate clearly-labeled synthetic order data for a reproducible demo."""
import numpy as np
import pandas as pd

RANDOM_STATE = 42

def make_dataset(n_rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    categories = np.array(["Clothing", "Electronics", "Home", "Beauty", "Footwear"])
    payment_methods = np.array(["Card", "UPI", "Cash on Delivery", "Wallet"])
    shipping_types = np.array(["Standard", "Express"])

    age = rng.integers(18, 66, n_rows)
    previous_purchases = rng.integers(0, 31, n_rows)
    historical_return_rate = rng.uniform(0, 0.8, n_rows)
    product_category = rng.choice(categories, n_rows)
    price = np.round(rng.uniform(10, 1200, n_rows), 2)
    discount_percent = rng.integers(0, 61, n_rows)
    product_rating = np.round(rng.uniform(1, 5, n_rows), 1)
    payment_method = rng.choice(payment_methods, n_rows)
    product_views = rng.integers(1, 31, n_rows)
    session_minutes = np.round(rng.uniform(1, 45, n_rows), 1)
    shipping_type = rng.choice(shipping_types, n_rows)
    used_coupon = rng.integers(0, 2, n_rows)

    # Synthetic relationship for demonstration only; not learned from real transactions.
    score = (
        -2.0
        + 3.0 * historical_return_rate
        + 0.018 * discount_percent
        + 0.30 * (product_category == "Clothing")
        + 0.45 * (product_category == "Footwear")
        + 0.25 * (product_rating < 3)
        + 0.30 * (payment_method == "Cash on Delivery")
        + 0.25 * used_coupon
        - 0.025 * np.minimum(previous_purchases, 20)
        - 0.08 * (shipping_type == "Express")
    )
    probability = 1 / (1 + np.exp(-score))
    returned = rng.binomial(1, probability)

    return pd.DataFrame({
        "age": age,
        "previous_purchases": previous_purchases,
        "historical_return_rate": historical_return_rate.round(3),
        "product_category": product_category,
        "price": price,
        "discount_percent": discount_percent,
        "product_rating": product_rating,
        "payment_method": payment_method,
        "product_views": product_views,
        "session_minutes": session_minutes,
        "shipping_type": shipping_type,
        "used_coupon": used_coupon,
        "returned": returned,
    })

if __name__ == "__main__":
    df = make_dataset()
    df.to_csv("data/synthetic_orders.csv", index=False)
    print(f"Created data/synthetic_orders.csv with {len(df)} synthetic rows.")
