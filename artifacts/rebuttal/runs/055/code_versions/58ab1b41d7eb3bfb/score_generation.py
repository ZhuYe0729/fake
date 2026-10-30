"""Merge complete, non-overlapping shards and score each original corpus once."""
import argparse
import json

from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu

from common import RUN, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', required=True)
    parser.add_argument('--task', choices=['cnn_dm_1000', 'dsum', 'IWSLT'], required=True)
    parser.add_argument('--results', nargs='+', required=True)
    args = parser.parse_args()
    expected = [json.loads(line) for line in (RUN / f'data/generation_{args.task}.jsonl').read_text().splitlines()]
    rows = []
    for filename in args.results:
        from pathlib import Path
        result = json.loads(Path(filename).read_text())
        assert result['complete'] and result['policy'] == args.policy and result['task'] == args.task
        rows.extend(json.loads(line) for line in Path(result['answers']).read_text().splitlines())
    rows.sort(key=lambda row: row['question_index'])
    assert [row['question_index'] for row in rows] == list(range(len(expected)))
    assert [row['question_id'] for row in rows] == [row['question_id'] for row in expected]
    assert [row['reference'] for row in rows] == [row['reference'] for row in expected]
    predictions = [row['prediction'].strip() or ' ' for row in rows]
    references = [row['reference'] for row in rows]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    metrics = {'complete': True, 'policy': args.policy, 'task': args.task, 'samples': len(rows),
        'rougeL_percent': 100 * sum(scorer.score(reference, prediction)['rougeL'].fmeasure
            for reference, prediction in zip(references, predictions, strict=True)) / len(rows),
        'empty_predictions': sum(not row['prediction'].strip() for row in rows), 'source_results': args.results}
    if args.task == 'IWSLT':
        metrics['sacre_bleu'] = corpus_bleu(predictions, [references]).score
    output = RUN / f'results/task_metrics/{args.policy}/{args.task}.json'
    assert not output.exists()
    write_json(output, metrics)


if __name__ == '__main__':
    main()
