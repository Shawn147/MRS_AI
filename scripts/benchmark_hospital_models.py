"""Separate task-specific research benchmarks; these models are never used in chat."""
import hashlib
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                            brier_score_loss, classification_report, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.regional_data import load_regional_preview, REGIONAL_DIR

TASKS = {
    'pakistan_heart_failure': {'positive': '1', 'drop': ['time'],
        'numeric': 'age anaemia creatinine_phosphokinase diabetes ejection_fraction high_blood_pressure platelets serum_creatinine serum_sodium sex smoking'.split(),
        'limitation': 'Death during variable follow-up is a retrospective binary study outcome, not a calibrated fixed-horizon survival forecast. Follow-up time is excluded as leakage.'},
    'bangladesh_sylhet_diabetes': {'positive': 'Positive', 'drop': [], 'numeric': ['age'],
        'limitation': 'Questionnaire study labels from one hospital; no independent external cohort. Exact duplicate observations are removed before splitting.'},
    'india_chronic_kidney_disease': {'positive': 'ckd', 'drop': [],
        'numeric': 'age bp sg al su bgr bu sc sod pot hemo pcv wbcc rbcc'.split(),
        'limitation': 'Laboratory features may also have informed the clinical label. Within-cohort classification does not establish prospective screening performance.'}}
OUT = ROOT / 'artifacts/research'


def preprocessing(numeric, categorical):
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical)],
        remainder='drop')


def evaluate(y, scores):
    predicted = (scores >= .5).astype(int)
    return {'accuracy': float(accuracy_score(y, predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'roc_auc': float(roc_auc_score(y, scores)),
            'average_precision': float(average_precision_score(y, scores)),
            'brier_score': float(brier_score_loss(y, scores)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
            'report': classification_report(y, predicted, labels=[0, 1], output_dict=True, zero_division=0),
            'test_rows': len(y)}


def main():
    catalog, manifest, audit = load_regional_preview()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for source in catalog['sources']:
        ident = source['id']
        if ident not in TASKS:
            continue
        settings = TASKS[ident]
        raw = (REGIONAL_DIR / (ident + '.jsonl')).read_bytes()
        rows = [json.loads(line) for line in raw.splitlines()]
        groups = [r['feature_group_id'] for r in rows]
        if len(set(groups)) != len(groups):
            raise ValueError('Duplicate feature groups must be resolved before benchmarking')
        features = [f for f in source['features'] if f not in settings['drop']]
        numeric = settings['numeric']
        categorical = [f for f in features if f not in numeric]
        x = pd.DataFrame([r['features'] for r in rows])[features]
        for col in numeric:
            x[col] = pd.to_numeric(x[col], errors='coerce')
        for col in categorical:
            x[col] = x[col].replace({None: np.nan})
        target = source['targets'][0]
        y = np.array([int(r['target'][target] == settings['positive']) for r in rows])
        indices = np.arange(len(rows))
        trainval, test = train_test_split(indices, test_size=.2, stratify=y, random_state=42)
        train, val = train_test_split(trainval, test_size=.25, stratify=y[trainval], random_state=42)
        parts = {'train': train, 'validation': val, 'test': test}
        splits = {k: [rows[i]['id'] for i in v] for k, v in parts.items()}
        candidates = [('logistic_C_' + str(c), LogisticRegression(C=c, max_iter=2000, class_weight='balanced', random_state=42)) for c in [.1, 1., 10.]]
        candidates += [('forest_depth_' + str(depth), RandomForestClassifier(n_estimators=150, max_depth=depth, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=2)) for depth in [6, None]]
        scored = []
        for name, classifier in candidates:
            pipeline = Pipeline([('prepare', preprocessing(numeric, categorical)), ('classifier', classifier)])
            pipeline.fit(x.iloc[train], y[train])
            score = balanced_accuracy_score(y[val], pipeline.predict(x.iloc[val]))
            scored.append((float(score), name, pipeline))
        validation_score, selected_name, selected = max(scored, key=lambda r: r[0])
        # No refit or tuning using test rows. Preprocessing is fitted on train only.
        scores = selected.predict_proba(x.iloc[test])[:, 1]
        dummy = DummyClassifier(strategy='prior').fit(np.zeros((len(train), 1)), y[train])
        baseline_scores = dummy.predict_proba(np.zeros((len(test), 1)))[:, 1]
        joblib.dump(selected, OUT / (ident + '.joblib'))
        item = {'source_id': ident, 'name': source['name'], 'country': source['country'],
                'task': source['task'], 'target': target, 'positive_label': settings['positive'],
                'source_url': source['source_url'], 'input_sha256': hashlib.sha256(raw).hexdigest(),
                'features': features, 'excluded_features': settings['drop'],
                'split_sizes': {k: len(v) for k, v in splits.items()}, 'splits': splits,
                'selected_model': selected_name, 'validation_balanced_accuracy': validation_score,
                'selection_candidates': [{'model': n, 'validation_balanced_accuracy': s} for s, n, _ in scored],
                'test': evaluate(y[test], scores), 'majority_baseline': evaluate(y[test], baseline_scores),
                'limitations': [settings['limitation'], 'Single-source internal holdout; not independent external or prospective validation.',
                                'No verified patient identifiers: distinct feature groups do not prove distinct patients.',
                                'Research model only; not used for chatbot diagnosis, triage or treatment.']}
        results.append(item)
        print(ident, selected_name, item['test']['balanced_accuracy'], flush=True)
    report = {'schema_version': 1, 'seed': 42, 'purpose': 'Task-specific public-hospital research benchmarks; not chat inference',
              'selection_policy': 'Stratified 60/20/20 after deduplication; fit preprocessing on train only; select by validation balanced accuracy; evaluate test once',
              'classifier_data_unchanged': True, 'models': results,
              'excluded_cohorts': [{'source_id': 'pakistan_yicss', 'reason': 'Clinical code meanings and screening-to-referral timing require review before modelling.'}]}
    (OUT / 'metrics.json').write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
