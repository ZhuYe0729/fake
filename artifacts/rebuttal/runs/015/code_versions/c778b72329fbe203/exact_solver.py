"""One-hot MILP, forward/inverse objectives on the identical additive surrogate."""
import time

import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import lil_matrix


def solve(q,t,cap,inverse=False):
    q=np.asarray(q,dtype=float);t=np.asarray(t,dtype=float)
    assert q.shape==t.shape and q.ndim==2
    n,k=q.shape;matrix=lil_matrix((n+1,n*k))
    for i in range(n):matrix[i,i*k:(i+1)*k]=1
    costs=(t*1000 if inverse else q*1e8).reshape(-1)
    matrix[-1]=costs;limit=cap*(1000 if inverse else 1e8)
    objective=(q*1e8 if inverse else t).reshape(-1)
    start=time.perf_counter()
    result=milp(objective,integrality=np.ones(n*k),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix.tocsr(),np.r_[np.ones(n),-np.inf],np.r_[np.ones(n),limit]),
        options={'mip_rel_gap':0.0,'time_limit':120.})
    elapsed=time.perf_counter()-start
    if result.x is None:raise RuntimeError(result.message)
    choice=np.rint(result.x).astype(int).reshape(n,k)
    assert np.all(choice.sum(1)==1)
    quality=float((q*choice).sum());latency=float((t*choice).sum())
    assert (latency if inverse else quality)<=cap+max(1e-9,abs(cap)*1e-8)
    return {'indices':choice.argmax(1).tolist(),'quality':quality,'latency_ms':latency,
        'seconds':elapsed,'status':int(result.status),'message':result.message,
        'optimal':bool(result.status==0),'mip_gap':float(result.mip_gap),'nodes':int(result.mip_node_count)}
