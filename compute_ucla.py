import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------
EXCEL_PATH = "./artifacts/ucla/ucla_responses.xlsx"
OUT_CSV    = "./artifacts/ucla/ucla_totals.csv"



# ---------------------------------------------------------------------
# 1. Load Excel
# ---------------------------------------------------------------------
df = pd.read_excel(EXCEL_PATH)

print("Loaded shape:", df.shape)
print("First columns:", df.columns[:8].tolist())
# print(df.columns.tolist())  # uncomment to inspect all headers

# ---------------------------------------------------------------------
# 2. Define UCLA items and scoring
# ---------------------------------------------------------------------
score_map = {
    "Never": 1,
    "Rarely": 2,
    "Sometimes": 3,
    "Often": 4,
}

item_stems = [
    "1.I feel in tune with the people around me. (R)",
    "2.I lack companionship.",
    "3. There is no one I can turn to.",
    "4. I feel alone.",
    "5. I feel part of a group of friends. (R)",
    "6. I have a lot in common with the people around me. (R)",
    "7. I am no longer close to anyone.",
    "8. My interests and ideas are not shared by those around me.",
    "9. I am an outgoing person. (R)",
    "10. There are people I feel close to. (R)",
    "11. I feel left out.",
    "12. My social relationships are superficial.",
    "13. No one really knows me well.",
    "14. I feel isolated from others.",
    "15. I can find companionship when I want it. (R)",
    "16. There are people who really understand me. (R)",
    "17. I am unhappy being so withdrawn.",
    "18. People are around me but not with me.",
    "19. There are people I can talk to. (R)",
    "20. There are people I can turn to. (R)",
]

reverse_items = {1, 5, 6, 9, 10, 15, 16, 19, 20}

# ---------------------------------------------------------------------
# 3. Find PRE and POST columns
# ---------------------------------------------------------------------
pre_cols = []
post_cols = []

for stem in item_stems:
    matches = [c for c in df.columns if str(c).startswith(stem)]
    if len(matches) == 1:
        pre_cols.append(matches[0])
        post_cols.append(None)
    elif len(matches) >= 2:
        pre_cols.append(matches[0])   # first = pre
        post_cols.append(matches[1])  # second = post
    else:
        print(f"WARNING: no column found for stem: {stem}")

print("Pre columns:", pre_cols)
print("Post columns:", post_cols)

# ---------------------------------------------------------------------
# 4. Convert text → numeric
# ---------------------------------------------------------------------
pre_df = df[pre_cols].replace(score_map)

valid_post_cols = [c for c in post_cols if c is not None]
post_df = df[valid_post_cols].replace(score_map)

# ---------------------------------------------------------------------
# 5. Apply reverse scoring
# ---------------------------------------------------------------------
for idx, stem in enumerate(item_stems, start=1):
    if idx in reverse_items:
        pre_col = pre_cols[idx - 1]
        pre_df[pre_col] = pre_df[pre_col].apply(
            lambda x: 5 - x if pd.notna(x) else x
        )

        post_col = post_cols[idx - 1]
        if post_col is not None:
            post_df[post_col] = post_df[post_col].apply(
                lambda x: 5 - x if pd.notna(x) else x
            )

# ---------------------------------------------------------------------
# 6. Compute totals and assemble result table
# ---------------------------------------------------------------------
df["UCLA_pre_total"]  = pre_df.sum(axis=1, min_count=1)
df["UCLA_post_total"] = post_df.sum(axis=1, min_count=1)
df["UCLA_delta"]      = df["UCLA_post_total"] - df["UCLA_pre_total"]

# choose an ID column if available; else fall back to Timestamp
id_col_candidates = [c for c in df.columns if "Session_ID" in str(c)]
id_col = id_col_candidates[0] if id_col_candidates else "Timestamp"

result = df[[id_col, "UCLA_pre_total", "UCLA_post_total", "UCLA_delta"]].copy()
result.rename(
    columns={
        id_col: "participant_id",
    },
    inplace=True,
)

print("\n=== UCLA totals per participant ===")
print(result)

# ---------------------------------------------------------------------
# 7. Save to CSV
# ---------------------------------------------------------------------
result.to_csv(OUT_CSV, index=False)
print(f"\nSaved per-participant totals to: {OUT_CSV}")

# ---------------------------------------------------------------------
# 8. Plot pre–post “spaghetti plot”
# ---------------------------------------------------------------------
# give a simple numeric index if you want it for plotting/inspection
if "participant" not in result.columns:
    result["participant"] = range(1, len(result) + 1)

fig, ax = plt.subplots(figsize=(6, 4))

for _, row in result.iterrows():
    ax.plot(
        ["Pre", "Post"],
        [row["UCLA_pre_total"], row["UCLA_post_total"]],
        marker="o",
        linewidth=1.5,
        alpha=0.7,
    )

pre_mean = result["UCLA_pre_total"].mean()
post_mean = result["UCLA_post_total"].mean()
ax.plot(
    ["Pre", "Post"],
    [pre_mean, post_mean],
    marker="o",
    linewidth=3,
    linestyle="--",
    label=f"Mean (Δ = {post_mean - pre_mean:.2f})",
)

ax.set_ylabel("UCLA Loneliness total score")
ax.set_title("Pre–Post UCLA Loneliness Scores per Participant")
ax.set_ylim(20, 80)  # theoretical range
ax.legend(loc="best")


plt.tight_layout()
OUT_PNG = "./artifacts/ucla/ucla_pre_post.png"
plt.savefig(OUT_PNG, dpi=300)
print(f"Saved pre–post plot to: {OUT_PNG}")
