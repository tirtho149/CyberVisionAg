from datasets import load_dataset

# Load dataset
ds = load_dataset("risashinoda/AgroBench")
df = ds["train"].to_pandas()

# Keep only questions containing the word "disease"
df_disease = df[df["question"].str.contains("disease", case=False, na=False)]

# Open output file
f = open("disease_question_pairs.txt", "w")

def log(text):
    print(text)          # real-time terminal output
    f.write(text + "\n") # save to file

log("Filtering questions containing the word 'disease'")
log(f"Total rows kept: {len(df_disease)}\n")

# Count crop-disease pairs
pair_counts = df_disease.groupby(["crop", "answer"]).size().reset_index(name="count")

log(f"Total unique crop-disease pairs: {len(pair_counts)}\n")

log("===== CROP | DISEASE | COUNT =====")
for _, row in pair_counts.iterrows():
    log(f"{row['crop']} | {row['answer']} | {row['count']}")

f.close()
print("\nSaved to disease_question_pairs.txt")