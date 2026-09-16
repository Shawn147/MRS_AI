"""Train with deduplicated 60/20/20 splits; tune only on train/validation, then evaluate test."""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from safetensors.torch import load_file, save_file
from transformers import AutoModel, AutoTokenizer
from src.data import ARTIFACT_DIR, fingerprint, load_data, symptom_text
from src.models import BASE_MODEL, BASE_REVISION, MAX_LENGTH, ConditionTransformer


def evaluate(actual, probabilities, labels):
    predicted = np.array(labels)[np.argmax(probabilities, axis=1)]
    return {'accuracy': float(accuracy_score(actual, predicted)),
            'report': classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(actual, predicted, labels=labels).tolist(),
            'test_rows': len(actual)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    data = load_data()
    rows = data['training']
    labels = sorted({r['condition'] for r in rows})
    indices = np.arange(len(rows))
    target = np.array([r['condition'] for r in rows])
    trainval, test = train_test_split(indices, test_size=0.2, stratify=target, random_state=42)
    train, val = train_test_split(trainval, test_size=0.25, stratify=target[trainval], random_state=42)
    assert not (set(train) & set(test) or set(train) & set(val) or set(val) & set(test))
    texts = [symptom_text(r['symptoms']) for r in rows]
    out = ARTIFACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    model_dir = out/'transformer'
    model_dir.mkdir(exist_ok=True)
    # Invalidate the completion marker while a new training run is in progress.
    if (out/'metrics.json').exists():
        (out/'metrics.json').replace(out/'previous_metrics.json')
    splits = {name: [rows[i]['id'] for i in part] for name, part in [('train',train),('validation',val),('test',test)]}
    (out/'splits.json').write_text(json.dumps(splits, indent=2)+'\n')
    print('SPLITS', {k:len(v) for k,v in splits.items()}, flush=True)
    fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    baseline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2))),
                         ('clf', LogisticRegression(max_iter=2000, random_state=42))])
    start = time.perf_counter()
    search = GridSearchCV(baseline, {'clf__C':[1.0,10.0,100.0]}, cv=fold, scoring='f1_macro')
    search.fit([texts[i] for i in train], target[train])
    baseline = search.best_estimator_
    baseline_seconds = time.perf_counter()-start
    joblib.dump(baseline, out/'baseline.joblib')
    pretrained = ROOT/'artifacts/pretrained/minilm'
    if not (pretrained/'model.safetensors').exists():
        raise FileNotFoundError('Run python scripts/download_model.py first to download the pretrained encoder.')
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True)
    encoder = AutoModel.from_pretrained(pretrained, local_files_only=True, use_safetensors=True)
    model = ConditionTransformer(encoder, len(labels))
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')
    y = torch.tensor([labels.index(t) for t in target], dtype=torch.long)
    def batch(part):
        return {k:v[part] for k,v in encoded.items()}
    def probabilities(part):
        model.eval()
        with torch.inference_mode():
            return torch.cat([model(batch(part[i:i+16])).softmax(-1) for i in range(0,len(part),16)]).numpy()
    start = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        embeddings = torch.cat([model.embed(batch(indices[i:i+16])) for i in range(0,len(rows),16)]).numpy()
    # A pretrained frozen encoder has seen no project labels. Head tuning uses training folds only.
    head_search = GridSearchCV(LogisticRegression(max_iter=2000, random_state=42), {'C':[1.,10.,100.]},
                               cv=fold, scoring='f1_macro')
    head_search.fit(embeddings[train], target[train])
    head = head_search.best_estimator_
    joblib.dump(head, out/'frozen_head.joblib')
    assert list(head.classes_) == labels
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor(head.coef_, dtype=torch.float32))
        model.head.bias.copy_(torch.tensor(head.intercept_, dtype=torch.float32))
    frozen_val = float(accuracy_score(target[val], head.predict(embeddings[val])))
    optimizer = torch.optim.AdamW([{'params':model.encoder.parameters(), 'lr':args.learning_rate},
                                   {'params':model.head.parameters(), 'lr':1e-3}], weight_decay=0.01)
    best_score, patience, history = None, 0, []
    for epoch in range(1,args.epochs+1):
        model.train()
        total_loss = 0.
        shuffled = np.random.permutation(train)
        for offset in range(0,len(shuffled),16):
            part = shuffled[offset:offset+16]
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(batch(part)), y[part])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            total_loss += float(loss.detach())*len(part)
        vp = probabilities(val)
        va = float(accuracy_score(target[val], np.array(labels)[vp.argmax(1)]))
        vl = float(log_loss(target[val], vp, labels=labels))
        item = {'epoch':epoch,'train_loss':total_loss/len(train),'validation_accuracy':va,'validation_loss':vl}
        history.append(item)
        print(json.dumps(item),flush=True)
        score = (va,-vl)
        if best_score is None or score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(model_dir/'model.safetensors'))
        else:
            patience += 1
        if patience >= 3:
            break
    training_seconds = time.perf_counter()-start
    model.load_state_dict(load_file(str(model_dir/'model.safetensors')))
    model.eval()
    encoder.config.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    # The held-out test set is evaluated only after training/selection is finished.
    metrics = {'schema_version':1,'data_fingerprint':fingerprint(),'labels':labels,
        'source_revision':data['manifest']['revision'], 'base_model':BASE_MODEL,'base_revision':BASE_REVISION,
        'seed':42, 'split_sizes':{k:len(v) for k,v in splits.items()},
        'raw_rows':data['manifest']['raw_rows'],'unique_rows':len(rows),
        'training':{'epochs_requested':args.epochs,'best_epoch':best_epoch,'learning_rate':args.learning_rate,
                    'head_learning_rate':1e-3,'batch_size':16,'max_length':MAX_LENGTH,
                    'encoder_fine_tuned':True,'seconds':training_seconds,'history':history},
        'optimization':{'baseline_best_C':search.best_params_['clf__C'],
                        'baseline_cv_macro_f1':float(search.best_score_),
                        'frozen_head_best_C':head_search.best_params_['C'],
                        'frozen_validation_accuracy':frozen_val,
                        'transformer_validation_accuracy':best_score[0]},
        'models':{'ml':evaluate(target[test],baseline.predict_proba([texts[i] for i in test]),labels),
                  'frozen':evaluate(target[test],head.predict_proba(embeddings[test]),labels),
                  'transformer':evaluate(target[test],probabilities(test),labels)}}
    metrics['models']['ml']['training_seconds'] = baseline_seconds
    for key in ['ml','transformer']:
        start = time.perf_counter()
        if key == 'ml': baseline.predict_proba([texts[test[0]]])
        else: probabilities(test[:1])
        metrics['models'][key]['inference_ms'] = (time.perf_counter()-start)*1000
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('TEST RESULTS', {k:v['accuracy'] for k,v in metrics['models'].items()},flush=True)

if __name__ == '__main__':
    main()
