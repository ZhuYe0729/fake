"""Restore the missing Chat checkpoint from its public pinned mirror, verify HF LFS."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi

RUN = Path(__file__).resolve().parents[1]
TARGET = Path('/dev/shm/daso_rebuttal_018_20261026/source_model')
REPO = 'shakechen/Llama-2-7b-chat-hf'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8*1024**2):
            value.update(block)
    return value.hexdigest()


def main():
    official = HfApi().model_info('meta-llama/Llama-2-7b-chat-hf', files_metadata=True)
    official_weights = {x.rfilename: x.lfs.sha256 for x in official.siblings
                        if x.rfilename.endswith('.safetensors')}
    url = f'https://modelscope.cn/api/v1/models/{REPO}/repo/files?Revision=master&Recursive=true'
    with urllib.request.urlopen(url, timeout=30) as response:
        listing = json.load(response)['Data']['Files']
    wanted = {'config.json','generation_config.json','tokenizer.model','tokenizer.json',
        'tokenizer_config.json','special_tokens_map.json','model.safetensors.index.json',
        'README.md','LICENSE.txt','USE_POLICY.md',*official_weights}
    files = [{key: x.get(key) for key in ['Name','Path','Size','Sha256','Revision']}
             for x in listing if x['Name'] in wanted]
    for row in files:
        if row['Name'] in official_weights:
            assert row['Sha256'] == official_weights[row['Name']], row['Name']
    assert sum(x['Name'] in official_weights for x in files) == 2
    TARGET.mkdir(parents=True, exist_ok=True)
    save(RUN / 'reference/source_model_download.json', {'official_repo': 'meta-llama/Llama-2-7b-chat-hf',
        'official_revision': official.sha, 'mirror_repo': REPO, 'files': files,
        'scope': 'Mirror weight SHA256 must exactly match official HF LFS identities; original empty model directory untouched.'})

    def download(row):
        path = TARGET / row['Name']
        status = RUN / 'results/download' / (row['Name']+'.json')
        if path.exists() and path.stat().st_size == row['Size'] and digest(path) == row['Sha256']:
            return row['Name']
        partial = path.with_suffix(path.suffix+'.partial')
        source = f'https://modelscope.cn/models/{REPO}/resolve/{row["Revision"]}/{row["Path"]}'
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(source, headers={'Range': f'bytes={offset}-'} if offset else {})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    if offset and response.status != 206:
                        offset = 0
                    with partial.open('ab' if offset else 'wb') as handle:
                        while data := response.read(8*1024**2):
                            handle.write(data)
                            offset += len(data)
                            save(status, {'complete': False, 'bytes': offset, 'total': row['Size'], 'attempt': attempt})
                assert partial.stat().st_size == row['Size'], row['Name']
                assert digest(partial) == row['Sha256'], row['Name']
                partial.replace(path)
                save(status, {'complete': True, 'bytes': row['Size'], 'sha256': row['Sha256']})
                print('Verified source asset', row['Name'], row['Size'], flush=True)
                return row['Name']
            except Exception as error:
                save(RUN / f'results/download_failures/{row["Name"]}_{time.time_ns()}.json',
                    {'attempt': attempt, 'bytes': offset, 'error': repr(error)})
                if attempt == 4:
                    raise
                time.sleep(10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert len(list(pool.map(download, sorted(files, key=lambda x:x['Size'])))) == len(files)
    spec = json.loads((RUN / 'configs/model.json').read_text())
    spec['original_missing_path'] = spec['model_path']
    spec['model_path'] = str(TARGET)
    save(RUN / 'configs/model.json', spec)
    save(RUN / 'results/source_model_complete.json', {'complete': True, 'path': str(TARGET),
        'official_revision': official.sha, 'weight_sha256': official_weights})


if __name__ == '__main__':
    main()
