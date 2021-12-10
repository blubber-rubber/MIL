import json
from collections import defaultdict

filename = 'temp.json'
header = [
    '| dataset | bag distance | int_dist | ext_owa | int_owa | Accuracy | F1 | TP | TN | FP | FN | Sensitivity | False Negative Rate | False Positive Rate | Specificity | Precission | False omission rate | FDR | Negative predictive value |\n',
    '|---------|--------------|----------|---------|---------|----------|----|----|----|----|----|-------------|---------------------|---------------------|-------------|------------|---------------------|-----|---------------------------|\n']

datasets = ['eastWest', 'elephant', 'fox', 'musk1', 'musk2', 'mutagenesis-atoms', 'mutagenesis-bonds',
            'mutagenesis-chains', 'tiger', 'westEast']


def sens(tp, fn):
    try:
        return str(round(tp / (tp + fn), 2))
    except:
        return 'Nan'


def fnr(fn, tp):
    try:
        return str(round(fn / (fn + tp), 2))
    except:
        return 'Nan'


def fpr(fp, tn):
    try:
        return str(round(fp / (fp + tn), 2))
    except:
        return 'Nan'


def spec(tn, fp):
    try:
        return str(round(tn / (tn + fp), 2))
    except:
        return 'Nan'


def prec(tp, fp):
    try:
        return str(round(tp / (tp + fp), 2))
    except:
        return 'Nan'


def npv(tn, fn):
    try:
        return str(round(tn / (fn + tn), 2))
    except:
        return 'Nan'


def fdr(fp, tp):
    try:
        return str(round(fp / (fp + tp), 2))
    except:
        return 'Nan'


def fomr(fn, tn):
    try:
        return str(round(fn / (fn + tn), 2))
    except:
        return 'Nan'


def accuracy(tp, fp, tn, fn):
    try:
        return str(round((tp + tn) / (tp + fp + fn + tn), 2))
    except:
        return 'Nan'


def f1(tp, fp, fn):
    try:
        return str(round((2 * tp) / (2 * tp + fp + fn), 2))
    except:
        return 'Nan'


with open(filename, 'r') as file:
    data = json.load(file)
    for dataset in datasets:
        rows = [result for result in data['results'] if result['data'] == dataset]
        table = []
        for result in rows:
            tp = result['TP']
            tn = result['TN']
            fp = result['FP']
            fn = result['FN']
            table.append(
                f'| {dataset} | {result["b_dist"]} | {result["int_dist"]} | {result["ext_owa"]} | {result["int_owa"]} | {accuracy(tp, fp, tn, fn)} | {f1(tp, fp, fn)} | {tp} | {tn} | {fp} | {fn} | {sens(tp, fn)} | {fnr(fn, tp)} | {fpr(fp, tn)} | {spec(tn, fp)} | {prec(tp, fp)} | {fomr(fn, tn)} | {fdr(fp, tp)} | {npv(tn, fn)} |\n')
        table.sort(key=lambda row: float(row.strip('|').split('|')[6].strip(' ')), reverse=True)
        f = open(f'results/{dataset}_results.md', 'w')
        f.write(''.join(header))
        f.write(''.join(table))
        f.close()
