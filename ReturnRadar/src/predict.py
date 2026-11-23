"""Predict return risk for one order from the command line."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "return_model.joblib"

def main():
    if not MODEL_PATH.exists():
        print("Model not found. First run: python src/train_model.py")
        return

    model = joblib.load(MODEL_PATH)
    print("\n--- ReturnRadar: Product Return Risk ---")
    print("Enter the order details below.\n")

    try:
        row = {
            "age": int(input("Customer age (e.g. 28): ")),
            "previous_purchases": int(input("Previous purchases (e.g. 5): ")),
            "historical_return_rate": float(input("Historical return rate, 0 to 1 (e.g. 0.20): ")),
            "product_category": input("Category [Clothing/Electronics/Home/Beauty/Footwear]: ").strip().title(),
            "price": float(input("Product price (e.g. 499): ")),
            "discount_percent": float(input("Discount percentage (e.g. 10): ")),
            "product_rating": float(input("Product rating, 1 to 5 (e.g. 4.2): ")),
            "payment_method": input("Payment [Card/UPI/Cash on Delivery/Wallet]: ").strip().title(),
            "product_views": int(input("Product views (e.g. 4): ")),
            "session_minutes": float(input("Session duration in minutes (e.g. 8): ")),
            "shipping_type": input("Shipping [Standard/Express]: ").strip().title(),
            "used_coupon": int(input("Used coupon? Enter 1 for yes, 0 for no: ")),
        }
        if not 0 <= row["historical_return_rate"] <= 1:
            raise ValueError("Historical return rate must be between 0 and 1.")
        if not 1 <= row["product_rating"] <= 5:
            raise ValueError("Product rating must be between 1 and 5.")
        if row["used_coupon"] not in (0, 1):
            raise ValueError("Coupon value must be 1 or 0.")

        order = pd.DataFrame([row])
        probability = model.predict_proba(order)[0, 1]
        risk = "High" if probability >= 0.65 else "Medium" if probability >= 0.35 else "Low"

        print("\n========== PREDICTION ==========")
        print(f"Estimated return probability: {probability:.1%}")
        print(f"Risk category: {risk}")
        print("This is a demo decision-support estimate, not a reason to deny a customer's return rights.")
        print("================================")
    except ValueError as error:
        print(f"\nInput error: {error}. Please run the script again and enter valid values.")

if __name__ == "__main__":
    main()
