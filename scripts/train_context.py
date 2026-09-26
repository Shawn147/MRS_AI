"""Train follow-up intent recognition; never use authored dialogue as clinical cases."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.context_model import DATASET, ARTIFACTS, embed


def main():
    dataset = json.loads(DATASET.read_text())
    rows = dataset['examples']
    texts = [r['text'] for r in rows]
    if len(set(t.strip().lower() for t in texts)) != len(texts):
        raise ValueError('Duplicate text across dataset splits')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate example IDs')
    parts = {s: np.array([i for i, r in enumerate(rows) if r['split'] == s]) for s in ['train', 'validation', 'test']}
    labels = np.array([r['intent'] for r in rows])
    x = embed(texts)
    candidates = []
    for regularization in [1., 10., 100.]:
        model = LogisticRegression(C=regularization, max_iter=2000, random_state=42).fit(x[parts['train']], labels[parts['train']])
        score = accuracy_score(labels[parts['validation']], model.predict(x[parts['validation']]))
        candidates.append((float(score), regularization, model))
    # Ties prefer the first (more regularized) model. Test set is untouched during selection.
    score, regularization, model = max(candidates, key=lambda item: item[0])
    predicted = model.predict(x[parts['test']])
    threshold = .60
    test_prob = model.predict_proba(x[parts['test']])
    sorted_prob = np.sort(test_prob, axis=1)
    accepted = (sorted_prob[:, -1] >= threshold) & ((sorted_prob[:, -1] - sorted_prob[:, -2]) >= .15)
    metrics = {'dataset_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
               'base_model': 'sentence-transformers/all-MiniLM-L6-v2', 'encoder_fine_tuned': False,
               'trained_component': 'multiclass logistic regression intent head',
               'split_sizes': {k: len(v) for k, v in parts.items()},
               'selected_C': regularization, 'validation_accuracy': score,
               'test_accuracy': float(accuracy_score(labels[parts['test']], predicted)),
               'test_report': classification_report(labels[parts['test']], predicted, output_dict=True, zero_division=0),
               'threshold': threshold, 'accepted_test_examples': int(accepted.sum()),
               'accepted_test_accuracy': float(accuracy_score(labels[parts['test']][accepted], predicted[accepted])) if accepted.any() else None,
               'limitations': dataset['provenance'],
               'splits': {k: [rows[i]['id'] for i in v] for k, v in parts.items()}}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / 'model.json').write_text(json.dumps({
        'dataset_sha256': metrics['dataset_sha256'], 'labels': model.classes_.tolist(),
        'weights': model.coef_.tolist(), 'intercepts': model.intercept_.tolist(), 'threshold': threshold}, indent=2)+'\n')
    (ARTIFACTS / 'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps({k: v for k, v in metrics.items() if k not in ['test_report', 'splits']}, indent=2))


if __name__ == '__main__':
    main()
