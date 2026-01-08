import os
import sys

def find_candidate_models(root_dir='.'):
    candidates = []
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if not os.path.isdir(path):
            continue
        has_model_file = any(
            os.path.exists(os.path.join(path, name))
            for name in ('pytorch_model.bin', 'model.safetensors', 'model.pt', 'tf_model.h5')
        )
        has_config = os.path.exists(os.path.join(path, 'config.json'))
        has_tokenizer = any(
            os.path.exists(os.path.join(path, t))
            for t in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json')
        )
        if has_model_file or (has_config and has_tokenizer):
            candidates.append(path)
    results_dir = os.path.join(root_dir, 'results')
    if os.path.isdir(results_dir):
        for entry in os.listdir(results_dir):
            p = os.path.join(results_dir, entry)
            if os.path.isdir(p):
                candidates.append(p)
    # dedupe
    return sorted(list(dict.fromkeys(candidates)))


def try_load_roberta():
    model_dir = 'roberta-base_ai_detector'
    if not os.path.isdir(model_dir):
        print('roberta-base_ai_detector not found')
        return 2
    try:
        import torch
        from transformers import pipeline
    except Exception as e:
        print('Missing dependency:', e)
        return 3

    device = 0 if torch.cuda.is_available() else -1
    print('Attempting to load model on', 'cuda' if device == 0 else 'cpu')
    try:
        clf = pipeline('text-classification', model=model_dir, tokenizer=model_dir, device=device)
        out = clf('This is a quick test sentence to validate the model.')
        print('PREDICTION:', out)
        return 0
    except Exception as e:
        print('Error loading/predicting with model:', e)
        return 4


if __name__ == '__main__':
    print('Candidate model directories:')
    for c in find_candidate_models('.'):
        print(' -', c)
    print('\nTrying to load roberta-base_ai_detector (if present)')
    rc = try_load_roberta()
    sys.exit(rc)
