"""The between-run reservation is never present during model initialization."""
import json
import os
import time
from pathlib import Path

from common import write_json


def handoff(operation,timeout=30,required=True):
    directory=os.environ.get('DASO_GPU_HANDOFF_DIR')
    if not directory:return None
    root=Path(directory);nonce=str(time.time_ns())
    write_json(root/'request.json',{'operation':operation,'nonce':nonce})
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if (root/'ack.json').exists():
            ack=json.loads((root/'ack.json').read_text())
            if ack['nonce']==nonce:return ack
        time.sleep(.05)
    if required:raise TimeoutError('GPU reservation handoff: '+operation)
    return None
