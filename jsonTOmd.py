import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from itertools import chain

feature_points = ["data", "b_dist", "int_dist", "int_owa", "picker", "pre_processors", "norm", "aggr"]

filename = 'results3.json'
header = [
    '| dataset | bag distance | int_dist | int_owa | picker | pre-processors | norm | Aggr | Accuracy | F1 | TP | TN | FP | FN |\n',
    '|---------|--------------|----------|---------|--------|----------------|------|------|----------|----|----|----|----|----|\n']

datasets = ['eastWest', 'elephant', 'fox', 'musk1', 'musk2', 'mutagenesis-atoms', 'mutagenesis-bonds',
            'mutagenesis-chains', 'tiger', 'westEast']

output_dir = 'default/'
import os

if not os.path.exists(f'results/{output_dir}'):
    os.makedirs(f'results/{output_dir}')

with open(filename, 'r') as file:
    data = json.load(file)
    for dataset in datasets:
        rows = [result for result in data['results'] if result['data'] == dataset]
        table = []
        for result in rows:
            predictions = [[int(i) for i in woord] for woord in result['predictions']]
            true = [[int(i) for i in woord] for woord in result['true']]
            # accuracy
            acc = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(true, predictions)]
            # F1
            fscore = [accuracy_score(y_true, y_pred) for y_true, y_pred in zip(true, predictions)]

            conf_matrix = confusion_matrix(list(chain(*true)), list(chain(*predictions)))
            tn, fp, fn, tp = conf_matrix.ravel()
            table.append(
                f'| {dataset} | {result["b_dist"]} | {result["int_dist"]}  | {result["int_owa"]} | {result["picker"]} | {result["pre_processors"]}  |  {result["norm"]} | {result["aggr"]} |{round(sum(acc) / len(acc), 3)} | {round(sum(fscore) / len(fscore), 3)}|{tp}|{tn}|{fp}|{fn}|\n')
        table.sort(key=lambda row: float(row.strip('|').split('|')[9].strip(' ')), reverse=True)
        f = open(f'results/{output_dir}{dataset}_results.md', 'w')
        f.write(''.join(header))
        f.write(''.join(table))
        f.close()
