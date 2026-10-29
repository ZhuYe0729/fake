"""Recover original SparseGPT sampling specifications for fair pruning controls."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT.parent/'runs'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import torch
    import pyarrow.ipc as ipc
    from transformers import AutoTokenizer
    source=next(Path('/root/wja/.cache/huggingface/datasets/Salesforce___wikitext').rglob('wikitext-train.arrow'))
    with source.open('rb') as handle:texts=ipc.open_stream(handle).read_all().column('text').to_pylist()
    text='\n\n'.join(texts)
    directory=ROOT/'pruning_calibration'
    directory.mkdir(exist_ok=True)
    for rid in ['017','018']:
        model=json.loads((RUNS/rid/'configs/model.json').read_text())
        metadata=Path(model['canonical_root'])/'sparse_bf16/metadata.json'
        spec=json.loads(metadata.read_text())['calibration']
        tokenizer=AutoTokenizer.from_pretrained(model['model_path'],local_files_only=True)
        ids=tokenizer(text,add_special_tokens=True)['input_ids']
        assert len(ids)==spec['token_count'],(rid,len(ids),spec['token_count'])
        tensor=torch.tensor([ids[start:start+spec['seq_len']] for start in spec['sample_starts']],dtype=torch.long)
        assert tuple(tensor.shape)==(128,2048)
        target=directory/f'{rid}_original_sg_128_2048.pt'
        assert not target.exists()
        torch.save(tensor,target)
        proof={'model':model['model_id'],'source_run':rid,'original_metadata':str(metadata),
            'metadata_sha256':sha(metadata),'arrow_source':str(source),'arrow_sha256':sha(source),
            'token_count':len(ids),'expected_token_count':spec['token_count'],'sample_starts':spec['sample_starts'],
            'samples':128,'sequence_length':2048,'add_special_tokens':True,'tensor':str(target),'sha256':sha(target),
            'scope':'Reconstructed from original recorded source, exact token count, seed/start list and model tokenizer. '
                    'Original calibration tensor was not retained; do not claim a byte-hash match to an unavailable original tensor.'}
        target.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n')
        print('Original SparseGPT sampling specification matched',rid,len(ids),tensor.shape,flush=True)


if __name__=='__main__':main()
