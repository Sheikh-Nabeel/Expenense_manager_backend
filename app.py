import os
from fastapi import FastAPI

from pydantic import BaseModel
from openai import OpenAI, RateLimitError, AuthenticationError, OpenAIError

# Load API keys from environment variables
API_KEYS = os.getenv("OPENAI_API_KEYS", "").split(",")
API_KEYS = [key.strip() for key in API_KEYS if key.strip()]

CATEGORIES = [
    "Food & Beverage", "Groceries", "Dining & Restaurants",
    "Transport", "Fuel", "Public Transport", "Ride Sharing",
    "Travel", "Flights", "Hotels & Accommodation",
    "Clothing", "Footwear", "Accessories",
    "Makeup", "Personal Care", "Skincare",
    "Health & Medical", "Pharmacy", "Fitness & Gym",
    "Electronics", "Home & Living",
    "Education", "Entertainment",
    "Shopping", "Financial Services",
    "Gifts & Donations",
    "Kids & Baby",
    "Pets",
    "Office & Work",
    "Sports & Outdoors",
    "Real Estate",
    "Legal & Government",
    "Other"
]

app = FastAPI()

# Input model
class Item(BaseModel):
    text: str

# API endpoint
@app.post("/classify")
def classify_item(item: Item):
    for key in API_KEYS:
        try:
            client = OpenAI(api_key=key)
            # 🔒 Force exact category name only
            prompt = f"""
Pick ONE category from the following list EXACTLY as it appears.
Do NOT add any extra text, punctuation, or explanation.
Categories: {CATEGORIES}
Item: "{item.text}"
Respond ONLY with the category name.
"""
            response = client.responses.create(
                model="gpt-4.1-mini",
                input=prompt
            )
            category = response.output[0].content[0].text.strip()
            # In case model adds quotes or extra spaces
            category = category.strip(' "\'')
            return {"category": category}
        except (RateLimitError, AuthenticationError, OpenAIError):
            continue
    return {"category": "Other"}
