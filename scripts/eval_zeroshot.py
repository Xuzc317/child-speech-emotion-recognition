"""Zero-shot evaluation: load checkpoint trained on source, evaluate on target."""
import argparse, json, os, sys, torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import get_cross_corpus_dataloaders
from src.train import SERModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            waveforms, labels, lengths, _, _ = batch
            waveforms, labels = waveforms.to(device), labels.to(device)
            lengths = lengths.to(device)
            logits = model(waveforms, lengths=lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    wa = accuracy_score(all_labels, all_preds)
    uar = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    return wa, uar

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--train_data', nargs='+', required=True)
    parser.add_argument('--test_data', nargs='+', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/logs')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get('config', {})
    config.setdefault('ssl_model', 'wavlm')
    config.setdefault('num_classes', 4)

    model = SERModel(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dls = get_cross_corpus_dataloaders(args.train_data, args.test_data, seed=args.seed)
    wa, uar = evaluate(model, dls['test'])
    test_n = len(dls['test'].dataset)

    os.makedirs(args.output, exist_ok=True)
    name = f"zeroshot_{'_'.join(args.train_data)}2{'_'.join(args.test_data)}"
    result = {
        'checkpoint': args.checkpoint,
        'train_data': args.train_data,
        'test_data': args.test_data,
        'seed': args.seed,
        'test_wa': float(wa),
        'test_uar': float(uar),
        'test_n': test_n,
        'type': 'zero_shot',
    }
    path = os.path.join(args.output, f'{name}.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Zero-shot: {args.train_data} -> {args.test_data}')
    print(f'WA={wa:.4f}, UAR={uar:.4f}, N={test_n}')
    print(f'Saved: {path}')

if __name__ == '__main__':
    main()
