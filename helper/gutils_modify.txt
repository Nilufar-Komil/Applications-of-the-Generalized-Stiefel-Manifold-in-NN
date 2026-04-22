import torch


def norm(v, dim=1):
    assert len(v.size()) == 2
    return v.norm(p=2, dim=dim, keepdim=True)


def unit(v, dim=1, eps=1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm


def xTy(x, y):
    assert len(x.size()) == 2 and len(y.size()) == 2, 'xTy'
    return torch.sum(x * y, dim=1, keepdim=True)


def clip_by_norm(v, clip_norm):
    v_norm = norm(v)
    scale = torch.ones_like(v_norm)
    mask = v_norm > clip_norm
    scale[mask] = clip_norm / v_norm[mask]
    return v * scale


def sym_matrix(y):  # y n-by-n
    assert y.size(0) == y.size(1)
    return (y + y.t()) / 2


def skew_matrix(y):  # y n-by-n
    assert y.size(0) == y.size(1)
    return (y - y.t()) / 2


def stiefel_proj_tan(y, g):  # y,g p-by-n, p <= n
    p, n = y.size()
    skew = skew_matrix(torch.matmul(y, g.t()))
    reflect = torch.matmul(y.t(), y)
    identity = torch.eye(n, device=y.device, dtype=y.dtype)
    reflect = identity - reflect
    tan_vec = torch.matmul(y.t(), skew) + torch.matmul(reflect, g.t())
    return tan_vec.t()


def stiefel_proj_norm(y, g):  # y,g p-by-n, p <= n
    sym = sym_matrix(torch.matmul(y, g.t()))
    norm_vec = torch.matmul(y.t(), sym)
    return norm_vec.t()


def polar_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
    p, n = tan_vec.size()
    U, S, Vh = torch.linalg.svd(tan_vec, full_matrices=False)
    return torch.matmul(U, Vh)


def qr_retraction(tan_vec):  # tan_vec, p-by-n, p <= n
    q, r = torch.linalg.qr(tan_vec.t(), mode='reduced')  # n-by-p
    d = torch.diagonal(r, 0)
    ph = d.sign()
    ph[ph == 0] = 1
    q = q * ph.unsqueeze(0).expand_as(q)
    return q.t()


def Cayley_loop(X, W, tan_vec, t):
    # X: n-by-p
    # W: n-by-n skew-symmetric
    # tan_vec: n-by-p
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()


def check_identity(X):  # X: n-by-p
    n, p = X.size()
    res = torch.eye(p, device=X.device, dtype=X.dtype) - torch.mm(X.t(), X)
    print('n={0}, p={1}, res norm={2}'.format(n, p, torch.norm(res)))


def stiefel_transport(y, g):  # y,g p-by-n, p <= n
    return stiefel_proj_tan(y, g)


def gproj(y, g, normalize=False):
    if normalize:
        y, _ = unit(y)

    yTg = xTy(y, g)
    return g - (yTg * y)


def gexp(y, h, normalize=False):
    if normalize:
        y, _ = unit(y)
        h = gproj(y, h)

    u, hnorm = unit(h)
    return y * hnorm.cos() + u * hnorm.sin()


# parallel translation of tangent vector h1 toward h2
# both h1 and h2 are tangent vectors on y
def gpt2(y, h1, h2, normalize=False):
    if normalize:
        h1 = gproj(y, h1)
        h2 = gproj(y, h2)

    u, unorm = unit(h2)
    uTh1 = xTy(u, h1)
    return h1 - uTh1 * (unorm.sin() * y + (1 - unorm.cos()) * u)


# parallel translation if h1 = h2
def gpt(y, h, normalize=False):
    if normalize:
        h = gproj(y, h)

    u, unorm = unit(h)
    return (u * unorm.cos() - y * unorm.sin()) * unorm
