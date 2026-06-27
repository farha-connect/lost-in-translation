import json

with open("translation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Keys:", data[0].keys())
print("\nFirst entry:")
for k, v in data[0].items():
    print(f"  {k}: {v}")