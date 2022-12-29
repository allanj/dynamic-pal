import pandas as pd
from collections import Counter
from pal.utils import read_data

"""
Analyze the performance of SSAT based on the problem types
"""

results = read_data('results/ssat_eval_result.json')
df = pd.read_csv('datasets/ssat_test_questions_no_image_shuffled_analysis.csv')


problem_type2corr = Counter()
problem_type2total = Counter()
## enumerate the data frame
for idx, row in df.iterrows():
    print(results[row["id"]]['score'])
    pt = row['Problem type']
    # if pt == 'Geometric':
    if row["Fine-grained Type"].strip() != '':
        problem_types = row["Fine-grained Type"].split(",")
        for problem_type in problem_types:
            problem_type2total[problem_type.strip()] += 1
            if results[row["id"]]['score'] == 1 and results[row["id"]]['answer'] != "":
                problem_type2corr[problem_type.strip()] += 1

# print(problem_type2corr)
# print(problem_type2total)
# # ## compute the accuracy for each problem type
# problem_type2acc = {}
# for problem_type in problem_type2total:
#     problem_type2acc[problem_type] = problem_type2corr[problem_type] / problem_type2total[problem_type] * 100
#
# ## print the accuracy
# print(problem_type2acc)