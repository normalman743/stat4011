import os
import csv

input_path = "/Users/mannormal/4011/Qi Zihan/v2/results/py/merge.csv"
output_dir = "/Users/mannormal/4011/Qi Zihan/v2/results/py/votes"
os.makedirs(output_dir, exist_ok=True)

with open(input_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for num in range(10):
    for inverse in [0, 1]:
        filename = f"vote_{num}_{inverse}.csv"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w", newline='', encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow(["ID", "Predict"])
            for row in rows:
                predict = float(row["Predict"])
                if inverse == 0:
                    vote = 0 if predict <= num else 1
                else:
                    vote = 1 if predict <= num else 0
                writer.writerow([row["ID"], vote])