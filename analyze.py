import json
import pandas as pd
from difflib import SequenceMatcher
import re

# Load data
with open("translation_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# --- Similarity Score ---
def similarity(a, b):
    a = a.lower().strip()
    b = b.lower().strip()
    return round(SequenceMatcher(None, a, b).ratio() * 100, 2)

df["similarity"] = df.apply(lambda r: similarity(r["original"], r["back_translated"]), axis=1)

# --- Word Count Change ---
df["original_words"] = df["original"].apply(lambda x: len(x.split()))
df["back_words"] = df["back_translated"].apply(lambda x: len(x.split()))
df["word_change_pct"] = round((df["back_words"] - df["original_words"]) / df["original_words"] * 100, 2)

# --- Punctuation Loss ---
def punct_count(text):
    return len(re.findall(r'[!?"\'\-;:()]', text))

df["orig_punct"] = df["original"].apply(punct_count)
df["back_punct"] = df["back_translated"].apply(punct_count)
df["punct_lost"] = df["orig_punct"] - df["back_punct"]

# --- Meaning Preservation Category ---
def meaning_label(sim):
    if sim >= 85: return "preserved"
    elif sim >= 60: return "partial_loss"
    else: return "significant_loss"

df["meaning_status"] = df["similarity"].apply(meaning_label)

# =====================
# PRINT ANALYSIS
# =====================

print("=" * 60)
print("LOST IN TRANSLATION - ANALYSIS REPORT")
print("=" * 60)

# 1. Overall stats
print(f"\nTotal translations: {len(df)}")
print(f"Average similarity: {df['similarity'].mean():.2f}%")
print(f"Median similarity: {df['similarity'].median():.2f}%")

# 2. By language
print("\n--- Similarity by Language ---")
print(df.groupby("target_lang")["similarity"].agg(["mean", "median", "min", "max"]).round(2).to_string())

# 3. By category
print("\n--- Similarity by Category ---")
print(df.groupby("category")["similarity"].agg(["mean", "median", "min", "max"]).round(2).to_string())

# 4. Language x Category
print("\n--- Language x Category (Mean Similarity) ---")
pivot = df.pivot_table(values="similarity", index="category", columns="target_lang", aggfunc="mean").round(2)
print(pivot.to_string())

# 5. Meaning preservation breakdown
print("\n--- Meaning Preservation ---")
print(df["meaning_status"].value_counts().to_string())

# 6. By language
print("\n--- Meaning Preservation by Language ---")
print(pd.crosstab(df["target_lang"], df["meaning_status"]).to_string())

# 7. Worst translations (most meaning lost)
print("\n--- Top 10 Worst Translations (Most Meaning Lost) ---")
worst = df.nsmallest(10, "similarity")[["category", "target_lang", "similarity", "original", "back_translated"]]
for _, row in worst.iterrows():
    print(f"\n[{row['category']}] [{row['target_lang']}] Similarity: {row['similarity']}%")
    print(f"  Original:    {row['original'][:80]}")
    print(f"  Back-trans:  {row['back_translated'][:80]}")

# 8. Word expansion/contraction
print("\n--- Avg Word Count Change % by Language ---")
print(df.groupby("target_lang")["word_change_pct"].mean().round(2).to_string())

# 9. Punctuation loss
print("\n--- Avg Punctuation Lost by Language ---")
print(df.groupby("target_lang")["punct_lost"].mean().round(2).to_string())

# 10. Save full results
df.to_csv("analysis_results.csv", index=False)
print("\n✅ Full results saved to analysis_results.csv")

# Save summary stats
summary = {
    "total": len(df),
    "avg_similarity": round(df["similarity"].mean(), 2),
    "by_language": df.groupby("target_lang")["similarity"].mean().round(2).to_dict(),
    "by_category": df.groupby("category")["similarity"].mean().round(2).to_dict(),
    "pivot": pivot.to_dict(),
    "meaning_counts": df["meaning_status"].value_counts().to_dict()
}

with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("✅ Summary saved to analysis_summary.json")