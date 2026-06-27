import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os

# ── Load pre-computed results from analyze.py ──────────────────────
df = pd.read_csv("analysis_results.csv")
print("✅ Loaded analysis_results.csv")

def meaning_label(sim):
    if sim >= 85:   return "Preserved"
    elif sim >= 60: return "Partial Loss"
    else:           return "Significant Loss"

df["meaning_status"] = df["meaning_status"] = df["similarity"].apply(meaning_label)

# ── Global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "sans-serif",
    "font.size"         : 11,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "grid.linestyle"    : "--",
    "figure.dpi"        : 130,
})

# ── Colour palette (two colors only: teal = good, red = loss) ──────
PRIMARY = "#0d7377"   # teal: preserved / good
DARK_PRIMARY = "#085f63"  # darker teal: mid language
DANGER  = "#c0392b"   # red: loss / risk
GOLD    = "#B8860B"   # only for average reference line
DARK    = "#1a1a2e"
MUTED   = "#7f8c8d"

LANG_NAMES = {"fr": "French", "hi": "Hindi", "ja": "Japanese"}
THRESHOLD  = 82

# ════════════════════════════════════════════════════════════════════
# CHART 1 — Average Meaning Preservation by Language
# ════════════════════════════════════════════════════════════════════
lang_means = df.groupby("target_lang")["similarity"].mean().sort_values()
overall_avg = df["similarity"].mean()

fig, ax = plt.subplots(figsize=(9, 5))

colors = [DANGER if v < THRESHOLD else PRIMARY for v in lang_means.values]
bars = ax.barh(
    [LANG_NAMES[l] for l in lang_means.index],
    lang_means.values,
    color=colors, edgecolor="white", height=0.55
)

ax.axvline(overall_avg, linestyle="--", color=GOLD, linewidth=2,
           label=f"Overall avg  {overall_avg:.1f}%")

for bar, val in zip(bars, lang_means.values):
    ax.text(bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", fontsize=9,
            color=DANGER if val < THRESHOLD else DARK)

ax.set_xlabel("Similarity Score (%)")
ax.set_title(
    "Average Meaning Preservation by Language\n"
    "Languages below the overall average highlighted in red",
    fontsize=13, fontweight="bold", pad=15
)
ax.set_xlim(70, 100)
ax.legend(
    handles=[
        Patch(facecolor=DANGER,  label="Below average: meaning at risk"),
        Patch(facecolor=PRIMARY, label="Above average"),
    ],
    loc="lower right", fontsize=9
)
plt.tight_layout()
plt.savefig("chart1_lang_similarity.png", bbox_inches="tight")
plt.close()
print("✅ chart1_lang_similarity.png saved")

# ════════════════════════════════════════════════════════════════════
# CHART 2 — Heatmap: Language × Category
# ════════════════════════════════════════════════════════════════════
pivot = df.pivot_table(
    values="similarity", index="category", columns="target_lang", aggfunc="mean"
).round(2)
pivot.columns = [LANG_NAMES[c] for c in pivot.columns]
pivot.index   = [c.capitalize() for c in pivot.index]

fig, ax = plt.subplots(figsize=(9, 5))

im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=70, vmax=95)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=11)

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val:.1f}%",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")

ax.set_title(
    "Meaning Preservation: Language × Category\n"
    "Red = high loss · Green = well preserved",
    fontsize=13, fontweight="bold", pad=15
)
plt.colorbar(im, ax=ax, label="Similarity %", shrink=0.8)
plt.tight_layout()
plt.savefig("chart2_heatmap.png", bbox_inches="tight")
plt.close()
print("✅ chart2_heatmap.png saved")

# ════════════════════════════════════════════════════════════════════
# CHART 3 — Meaning Preservation Breakdown (stacked bar)
# ════════════════════════════════════════════════════════════════════
status_colors = {
    "Preserved"        : PRIMARY,
    "Partial Loss"     : MUTED,
    "Significant Loss" : DANGER,
}

ct = pd.crosstab(df["target_lang"], df["meaning_status"])

for col in ["Preserved", "Partial Loss", "Significant Loss"]:
    if col not in ct.columns:
        ct[col] = 0

ct = ct[["Preserved", "Partial Loss", "Significant Loss"]]
ct.index = [LANG_NAMES[l] for l in ct.index]
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(10, 5))

bottom = np.zeros(len(ct_pct))
for status in ["Preserved", "Partial Loss", "Significant Loss"]:
    vals = ct_pct[status].values
    bars = ax.barh(ct_pct.index, vals, left=bottom,
                   color=status_colors[status], label=status,
                   height=0.55, edgecolor="white", linewidth=1.5)
    for bar, val, b in zip(bars, vals, bottom):
        if val > 6:
            ax.text(b + val / 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white")
    bottom += vals

ax.set_xlim(0, 100)
ax.set_xlabel("Percentage of Translations (%)")
ax.set_title(
    "Meaning Preservation Breakdown by Language\n"
    "Share of translations that fully preserved, partially lost, or significantly lost meaning",
    fontsize=13, fontweight="bold", pad=15
)
ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.14),
    ncol=3, fontsize=9, frameon=True
)
plt.tight_layout()
plt.savefig("chart3_meaning_breakdown.png", bbox_inches="tight")
plt.close()
print("✅ chart3_meaning_breakdown.png saved")

# ════════════════════════════════════════════════════════════════════
# CHART 4 — Meaning Preserved vs Lost (Stacked Horizontal Bar)
# ════════════════════════════════════════════════════════════════════
lang_means = df.groupby("target_lang")["similarity"].mean()
loss = (100 - lang_means).sort_values(ascending=False)
preserved = 100 - loss

languages = [LANG_NAMES[l] for l in loss.index]
loss_vals = loss.values
preserved_vals = preserved.values

fig, ax = plt.subplots(figsize=(10, 5))

bars_preserved = ax.barh(
    languages,
    preserved_vals,
    height=0.5,
    color=PRIMARY,
    label="Meaning Preserved",
    edgecolor="white",
    linewidth=1.5
)

bars_lost = ax.barh(
    languages,
    loss_vals,
    height=0.5,
    left=preserved_vals,
    color=DANGER,
    label="Meaning Lost",
    edgecolor="white",
    linewidth=1.5
)

for bar, val in zip(bars_preserved, preserved_vals):
    ax.text(
        bar.get_width() / 2,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}% preserved",
        va="center", ha="center",
        fontsize=11, fontweight="bold",
        color="white"
    )

for bar, val, left in zip(bars_lost, loss_vals, preserved_vals):
    ax.text(
        left + val / 2,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}% lost",
        va="center", ha="center",
        fontsize=11, fontweight="bold",
        color="white"
    )

ax.axvline(85, linestyle="--", color=GOLD,
           linewidth=1.5, label="85% quality benchmark")

ax.set_xlim(0, 100)
ax.set_xlabel("Percentage (%)", fontsize=11)
ax.set_title(
    "How Much Meaning Survives Translation?\n"
    "Share of meaning preserved vs lost per language",
    fontsize=13, fontweight="bold", pad=15
)

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=3,
    fontsize=9,
    frameon=True
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.savefig("chart4_meaning_lost.png", bbox_inches="tight")
plt.close()
print("✅ chart4_meaning_lost.png saved")

# ════════════════════════════════════════════════════════════════════
# CHART 5 — Translation Fidelity: Category × Language
# ════════════════════════════════════════════════════════════════════
cats  = df["category"].unique().tolist()
langs = ["fr", "hi", "ja"]
x     = np.arange(len(cats))
width = 0.25

lang_colors = {
    "fr": PRIMARY,        # teal: best performing
    "hi": DARK_PRIMARY,   # darker teal: mid
    "ja": DANGER,         # red: highest drift
}

fig, ax = plt.subplots(figsize=(10, 6))

for i, lang in enumerate(langs):
    vals = [
        df[(df["category"] == c) & (df["target_lang"] == lang)]["similarity"].mean()
        for c in cats
    ]
    bars = ax.bar(
        x + i * width, vals, width,
        label=LANG_NAMES[lang],
        color=lang_colors[lang],
        edgecolor="white", linewidth=1.5, alpha=0.9
    )
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.axhline(overall_avg, linestyle="--", color=GOLD, linewidth=1.5,
           label=f"Overall avg  {overall_avg:.1f}%")

ax.set_xticks(x + width)
ax.set_xticklabels([c.capitalize() for c in cats], fontsize=12)
ax.set_ylim(60, 105)
ax.set_ylabel("Similarity Score (%)")
ax.set_title(
    "Translation Fidelity: Category × Language\n"
    "Sarcasm and emotional content are consistently hardest to preserve",
    fontsize=13, fontweight="bold", pad=15
)
ax.legend(loc="upper left", fontsize=9, frameon=True)
plt.tight_layout()
plt.savefig("chart5_category_comparison.png", bbox_inches="tight")
plt.close()
print("✅ chart5_category_comparison.png saved")

print("\n🎉 All 5 charts saved!")