import requests

# USDA API Key (get one from https://fdc.nal.usda.gov/api-key-signup.html)
USDA_API_KEY = "YOUR_USDA_API_KEY"

def get_food_info(barcode: str):
    """
    Try to get food nutrition info by barcode.
    1. Query Open Food Facts (OFF) first.
    2. If not found, fallback to USDA FoodData Central (FDC).
    """

    # --- Step 1: Query Open Food Facts ---
    off_url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    off_res = requests.get(off_url).json()

    if off_res.get("status") == 1:
        product = off_res["product"]
        return {
            "source": "Open Food Facts",
            "product_name": product.get("product_name"),
            "brand": product.get("brands"),
            "nutriments": product.get("nutriments", {})
        }

    # --- Step 2: Fallback to USDA FoodData Central ---
    fdc_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": barcode,  # try search by barcode text (some branded foods include UPC)
        "pageSize": 1,
        "api_key": USDA_API_KEY
    }
    fdc_res = requests.get(fdc_url, params=params).json()

    if "foods" in fdc_res and len(fdc_res["foods"]) > 0:
        food = fdc_res["foods"][0]
        return {
            "source": "USDA FDC",
            "description": food.get("description"),
            "brand_owner": food.get("brandOwner"),
            "fdc_id": food.get("fdcId"),
            "nutrients": food.get("foodNutrients", [])
        }

    return {"error": "No data found in OFF or USDA"}

# --- Example usage ---
if __name__ == "__main__":
    barcode = "788434106832"  # Example: Oreo cookies UPC
    info = get_food_info(barcode)
    print(info)
