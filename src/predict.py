import json
import pickle
from src.data import ROOT, validate_sample

def load_artifacts(model_name=None):
    metadata=json.loads((ROOT/'models/metadata.json').read_text())
    name = model_name or metadata['winner']
    registry = metadata.get('models', {})
    if name not in registry:
        if model_name is not None:
            raise ValueError(f'Unknown model: {name}')
        filename = 'model.pkl'
    else:
        filename = registry[name]['file']
        metadata['test_metrics'] = registry[name]['test_metrics']
    metadata['active_model'] = name
    # Load only local artifacts named in the training-generated registry.
    with (ROOT/'models'/filename).open('rb') as f: model=pickle.load(f)
    return model,metadata

def predict(sample, model=None, metadata=None):
    if model is None or metadata is None: model,metadata=load_artifacts()
    frame,warnings=validate_sample(sample,metadata)
    value=float(model.predict(frame)[0])
    if value < 0: warnings.append('Model returned a negative estimate; this input is not supported reliably')
    return {'prediction':value,'unit':'LKR lakhs','prediction_lkr':value * 100000,'currency':'LKR','warnings':warnings,'model':metadata.get('active_model',metadata['winner'])}

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--input',help='JSON file containing raw feature values')
    parser.add_argument('--model',help='Model name, e.g. XGBoost or Random Forest')
    args=parser.parse_args()
    sample=json.loads(open(args.input).read()) if args.input else json.loads((ROOT/'models/demo.json').read_text())['input']
    print(json.dumps(predict(sample,*load_artifacts(args.model)),indent=2))
