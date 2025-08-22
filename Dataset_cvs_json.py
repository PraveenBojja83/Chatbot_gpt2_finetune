import pandas as pd
import json

df = pd.read_csv("Hotel_Dataset.csv")
json_data = [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]} for q, a in zip(df["Question"], df["Answer"])]

with open("hotel_chatbot_data.json", "w") as f:
    json.dump(json_data, f, indent=4)