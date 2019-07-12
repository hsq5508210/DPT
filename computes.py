from DPT import *
import numba
import cmath
import operator
from numba import vectorize, guvectorize, float32, cuda, jit
#==================================================================================
# Compute the innerproduct of the 'V'.
@guvectorize([(float32[:], float32[:])], '(n)->()', target = 'cuda')
def vectorProduce(V, res):
    for i in range(V.shape[0]):
        res[0] += V[i]*V[i]
#==================================================================================
# Compute the dot of the vector 'V1' and 'V2'.
@guvectorize([(float32[:], float32[:], float32[:])], '(n),(n)->()', target = 'cuda')
def dot(V1, V2, res):
    for i in range(V1.shape[0]):
        res[0] += V1[i]*V2[i]
# ==================================================================================
# Compute correlation between vector 'data' and 'label'.
@jit(float32(float32[:], float32[:]))
def Pierson(data, label):
    data = np.array(data, dtype='float32').reshape((-1, ))
    label = np.array(label, dtype='float32').reshape((-1, ))
    d = vectorProduce(data)[0]
    # print(d)
    d = cmath.sqrt(d)
    l = vectorProduce(label)[0]
    # print(l)
    l = cmath.sqrt(l)
    a = dot(data, label)[0]
    # print(d, l, a)
    return float(a/(d*l))
#==================================================================================

