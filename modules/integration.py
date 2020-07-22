import numpy as np
from tqdm import tqdm

def calculate_integration_weights(N=1000, M=1000):
    weights = np.zeros((32,32,32,32,2), dtype=float)
    count = np.zeros((32,32), dtype=int)
    for _ in tqdm(range(N)):
        x0, y0 = 16, 16
        S = np.zeros((32, 32, 2))
        for _ in range(M):
            dir = np.random.randint(4)
            if dir == 0:
                if x0 + 1 >= 32:
                    continue
                S[x0, y0, 0] += 1/2
                S[x0 + 1, y0, 0] += 1/2
                x0 += 1
            elif dir == 1:
                if y0 + 1 >= 32:
                    continue
                S[x0, y0, 1] += 1/2
                S[x0, y0 + 1, 1] += 1/2
                y0 += 1
            elif dir == 2:
                if x0 - 1 < 0:
                    continue
                S[x0, y0, 0] -= 1/2
                S[x0 - 1, y0, 0] -= 1/2
                x0 -= 1
            elif dir == 3:
                if y0 - 1 < 0:
                    continue
                S[x0, y0, 1] -= 1/2
                S[x0, y0 - 1, 1] -= 1/2
                y0 -= 1
            weights[x0, y0] += S
            count[x0,y0] += 1
    new_weights = np.zeros((32,32,32,32,2), dtype=float)
    for i in range(32):
        for j in range(32):
            new_weights[i,j] = weights[i,j] / count[i,j]
    return new_weights

def integrate(dx, dy, weights):
    f = np.zeros(dx.shape)
    for i in range(32):
        for j in range(32):
            f[:,i,j]=(dx*weights[i,j,:,:,0] + dy*weights[i,j,:,:,1]).sum((1,2))
    return f

def get_z(K, weights):
    '''
    K np array shape (*, 32, 32, 3)
    return np array shape (*, 32, 32, 1)
    '''
    from numpy import expand_dims
    return expand_dims(integrate(integrate(K[:,:,:,0], K[:,:,:,1], weights), integrate(K[:,:,:,1], K[:,:,:,2], weights), weights), -1)

def adapt_integration_constants(f, f_ref):
    '''
    f, f_ref: np array of shape (32, 32, 1)
    returns: np array of shape (32, 32, 1)
    '''
    from numpy.linalg import solve
    from numpy import cov, matrix
    x = np.array([[[x] for y in range(32)] for x in range(32)])
    y = np.array([[[y] for y in range(32)] for x in range(32)])
    X, Y, F, F_ref = [_.flatten() for _ in [x, y, f, f_ref]]
    diff = F_ref-F
    A = matrix([[X.var(), 0, 0], [0, Y.var(), 0], [X.mean(), Y.mean(), 1]])
    B = matrix([cov(diff, X)[0,1], cov(diff, Y)[0,1], diff.mean()]).T
    a, b, c = [_.item() for _ in solve(A, B)]
    f_new = f + a * x + b * y + c
    return f_new
