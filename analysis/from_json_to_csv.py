
from pal.utils import read_data, write_data
from pandas import json_normalize

data = read_data("results/ssat_eval_result.json")


new_data = []
for idx, obj in enumerate(data):
    if obj["score"] == 0:
        new_data.append({
            "id": idx,
            "question": obj["question"],
            "gold_answer": obj["answer"],
            "PAL prediction": "correct" if obj["score"] == 1 else "incorrect"})

df = json_normalize(new_data)
df.to_csv(f'datasets/ssat_test_questions_no_image_only_incorrect.csv', index=False)