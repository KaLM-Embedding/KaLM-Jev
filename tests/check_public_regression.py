"""Check the selected Nano setup against saved public JevBench predictions."""
import argparse
import hashlib
import json
from pathlib import Path

from kalm_jev import Engine, Request
from kalm_jev.compiler import compile_request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--reference-predictions', type=Path, required=True)
    parser.add_argument('--model-path', required=True)
    args = parser.parse_args()
    rows = [(split, json.loads(line)) for split in ('easy', 'original', 'hard')
            for line in (args.benchmark_root / 'datasets/public' / (split + '.jsonl')).read_text().splitlines()]
    prior = {row['task_id']: row for line in args.reference_predictions.read_text().splitlines()
             if (row := json.loads(line))}
    engine = Engine(model='kalm-jev-nano', model_path=args.model_path, device='cuda',
                    dtype='bfloat16', batch_size=4, query_max_length=8192,
                    decoder_max_length=9216, chunk_size=4)
    checks = []
    for i, (split, row) in enumerate(rows, 1):
        request = Request.model_validate({'state': row['state'], 'questions': {'decision': row['question']}})
        if row['question']['type'] == 'noul':
            assert [t.document for t in compile_request(request)] == [row['question']['criteria'][k] for k in ('true', 'false')]
        response = engine.evaluate(request)
        answer = response['answers']['decision']
        probs = {'yes': answer['noul'], 'no': 1 - answer['noul']} if row['question']['type'] == 'noul' else answer['probabilities']
        reference = prior[row['id']]
        predicted = max(sorted(probs), key=probs.get)
        checks.append({'task_id': row['id'], 'split': split, 'probabilities': probs,
                       'predicted': predicted, 'correct': predicted == str(row['expected']),
                       'prediction_matches': predicted == reference['predicted'],
                       'max_probability_diff': max(abs(probs[k] - reference['probs'][k]) for k in probs)})
        if i % 30 == 0:
            print(i, len(rows), flush=True)
    report = {'template_version': response['kalm']['template_version'], 'identity': engine.backend.public_identity,
              'configuration': {'dtype': 'bfloat16', 'batch_size': 4, 'chunk_size': 4,
                                'query_max_length': 8192, 'document_max_length': 1024, 'decoder_max_length': 9216},
              'n': len(checks), 'correct': sum(c['correct'] for c in checks),
              'prediction_mismatches': sum(not c['prediction_matches'] for c in checks),
              'max_probability_diff': max(c['max_probability_diff'] for c in checks),
              'reference_predictions_sha256': hashlib.sha256(args.reference_predictions.read_bytes()).hexdigest(),
              'note': 'All 74 Noul cases contain both criteria; this regression does not measure omitted-criteria behavior.',
              'cases': checks}
    (ROOT / 'results/noul-v3-nano-regression.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print({k: v for k, v in report.items() if k not in ('identity', 'cases')}, flush=True)
    assert report['n'] == 231 and report['prediction_mismatches'] == 0 and report['max_probability_diff'] < 1e-6


if __name__ == '__main__':
    main()
