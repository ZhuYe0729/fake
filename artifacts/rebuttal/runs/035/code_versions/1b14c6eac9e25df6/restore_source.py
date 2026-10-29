"""Restore exactly the already-verified018 Chat source, without new metadata APIs."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
TARGET = RUN / 'data/source_model'
MANIFEST = RUN.parent / '018/reference/source_model_download.json'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    spec=json.loads(MANIFEST.read_text())
    TARGET.mkdir(parents=True,exist_ok=True)

    def download(row):
        path=TARGET/row['Name'];partial=path.with_suffix(path.suffix+'.partial')
        status=RUN/'results/download'/(row['Name']+'.json')
        if path.exists() and path.stat().st_size==row['Size'] and digest(path)==row['Sha256']:
            return row['Name']
        source=f'https://modelscope.cn/models/{spec["mirror_repo"]}/resolve/{row["Revision"]}/{row["Path"]}'
        for attempt in range(5):
            offset=partial.stat().st_size if partial.exists() else 0
            try:
                request=urllib.request.Request(source,headers={'Range':f'bytes={offset}-'} if offset else {})
                with urllib.request.urlopen(request,timeout=30) as response:
                    if offset and response.status!=206:offset=0
                    with partial.open('ab' if offset else 'wb') as handle:
                        while data:=response.read(8*1024**2):
                            handle.write(data);offset+=len(data)
                            save(status,{'complete':False,'bytes':offset,'total':row['Size'],'attempt':attempt})
                assert partial.stat().st_size==row['Size'] and digest(partial)==row['Sha256'],row['Name']
                partial.replace(path)
                save(status,{'complete':True,'bytes':row['Size'],'sha256':row['Sha256']})
                print('Verified source',row['Name'],flush=True)
                return row['Name']
            except Exception as error:
                save(RUN/f'results/download_failures/{row["Name"]}_{time.time_ns()}.json',
                     {'error':repr(error),'attempt':attempt,'bytes':offset})
                if attempt==4:raise
                time.sleep(5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        assert len(list(pool.map(download,sorted(spec['files'],key=lambda r:r['Size']))))==len(spec['files'])
    save(RUN/'results/source_model_complete.json',{'complete':True,'path':str(TARGET),
        'source_manifest_sha256':digest(MANIFEST),'official_revision':spec['official_revision'],
        'files_sha256':{r['Name']:r['Sha256'] for r in spec['files']}})


if __name__=='__main__':main()
