import json
import time
import random
import os
from deep_translator import GoogleTranslator

with open("sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

languages = ["ja", "hi", "fr"]
results_file = "translations.json"

if os.path.exists(results_file):
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

done = set()
for r in results:
    done.add((r["original"], r["language"]))

tasks = []
for category, sents in sentences.items():
    for s in sents[:17]:
        for lang in languages:
            if (s, lang) not in done:
                tasks.append((category, s, lang))

total = len(tasks) + len(done)
completed = len(done)

print(f"Total translations needed: {total}")
print(f"Already done: {completed}")
print(f"Remaining: {len(tasks)}")
print("-" * 50)

for category, sentence, lang in tasks:
    completed += 1
    try:
        forward = GoogleTranslator(source="en", target=lang).translate(sentence)
        time.sleep(random.uniform(2, 4))

        back = GoogleTranslator(source=lang, target="en").translate(forward)
        time.sleep(random.uniform(2, 4))

        results.append({
            "category": category,
            "original": sentence,
            "target_lang": lang,          # fixed: was "language"
            "forward_translation": forward,
            "back_translated": back        # fixed: was "back_translation"
        })

        preview = sentence[:50] + "..." if len(sentence) > 50 else sentence
        print(f"[{completed}/{total}] {category} → {lang}: {preview}")

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[{completed}/{total}] FAILED {category} → {lang}: {e}")
        time.sleep(5)

print(f"\nDone! {len(results)} translations saved to {results_file}")