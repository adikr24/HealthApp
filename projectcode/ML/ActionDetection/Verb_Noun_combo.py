import pandas as pd

# -------------------------------
# Paths
# -------------------------------
train_csv = "/app/mediaFiles/videos/kinetic_400_label/EPIC_100_train.csv"
verb_csv = "/app/mediaFiles/videos/kinetic_400_label/EPIC_100_verb_classes.csv"
noun_csv = "/app/mediaFiles/videos/kinetic_400_label/EPIC_100_noun_classes.csv"
output_csv = "/app/mediaFiles/videos/kinetic_400_label/tsm_397_actions.csv"

# -------------------------------
# Load CSVs
# -------------------------------
train = pd.read_csv(train_csv)
verbs = pd.read_csv(verb_csv)
nouns = pd.read_csv(noun_csv)

# Count frequency of (verb_class, noun_class) pairs
top397 = (
    train.groupby(["verb_class", "noun_class"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .head(397)
    .reset_index(drop=True)
)

# Map IDs to text safely
verb_map = dict(zip(verbs["id"], verbs["key"]))
noun_map = dict(zip(nouns["id"], nouns["key"]))

top397["verb"] = top397["verb_class"].map(verb_map).fillna("unknown_verb")
top397["noun"] = top397["noun_class"].map(noun_map).fillna("unknown_noun")

# Merge verb + noun into single action
top397["action"] = top397["verb"] + " " + top397["noun"]

# Keep only id + action
top397["id"] = top397.index 
final_top397 = top397[["id", "action"]]

# Save CSV
final_top397.to_csv(output_csv, index=False)
print(f"✅ Created 397-action CSV at {output_csv}")
print(final_top397.head())
