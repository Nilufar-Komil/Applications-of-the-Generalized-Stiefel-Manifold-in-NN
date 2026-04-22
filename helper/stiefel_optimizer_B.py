import torch
from torch.optim.optimizer import Optimizer, required

from gutils_B import clip_by_norm
from gutils_B import stiefel_proj_tan
from gutils_B import qr_retraction
from gutils_modify import Cayley_loop
from utils_modify import matrix_norm_one

EPS32 = 1e-8


def _build_B_reduced_factors(B, eps=1e-10):
    """
    B: n x n symmetric PSD
    returns:
        U_r      : n x r
        evals_r  : r
        C        : n x r      = U_r diag(sqrt(lambda))
        C_t      : r x n      = C^T
        C_dag_t  : n x r      = U_r diag(1/sqrt(lambda))
        P_range  : n x n      = U_r U_r^T
    """
    evals, U = torch.linalg.eigh(B)
    keep = evals > eps
    if not torch.any(keep):
        raise ValueError("B has no positive eigenvalues above eps.")

    evals_r = evals[keep]
    U_r = U[:, keep]

    sqrt_e = torch.sqrt(evals_r)
    inv_sqrt_e = 1.0 / sqrt_e

    C = U_r * sqrt_e.unsqueeze(0)          # n x r
    C_t = C.transpose(0, 1)                # r x n
    C_dag_t = U_r * inv_sqrt_e.unsqueeze(0)  # n x r
    P_range = U_r @ U_r.transpose(0, 1)    # n x n

    return {
        "U_r": U_r,
        "evals_r": evals_r,
        "C": C,
        "C_t": C_t,
        "C_dag_t": C_dag_t,
        "P_range": P_range,
        "rank": int(evals_r.numel()),
    }


def _get_B_cache(group, device, dtype):
    """
    Cache the reduced factors once per parameter group/device/dtype.
    """
    B = group["B"]
    eps = group.get("eps", 1e-10)

    if B is None:
        raise ValueError("When stiefel=True, parameter group must provide B.")

    B_local = B.to(device=device, dtype=dtype)
    cache = group.get("_B_cache", None)

    if (
        cache is None
        or cache["device"] != device
        or cache["dtype"] != dtype
        or cache["shape"] != tuple(B_local.shape)
    ):
        fac = _build_B_reduced_factors(B_local, eps=eps)
        fac["device"] = device
        fac["dtype"] = dtype
        fac["shape"] = tuple(B_local.shape)
        group["_B_cache"] = fac

    return group["_B_cache"]


def _project_to_range(X, P_range):
    return P_range @ X


def _row_to_reduced(y, C_t):
    # y: p x n, X = y^T: n x p, Z = C^T X = C^T y^T
    return C_t @ y.transpose(0, 1)   # r x p


def _reduced_to_row(Z, C_dag_t):
    # X = C^{dagger,T} Z, y = X^T
    X = C_dag_t @ Z                  # n x p
    return X.transpose(0, 1)         # p x n


def _sequential_cayley(Z, W, V, alpha, s=2):
    """
    Same fixed-point idea as the original paper:
        Y^{(0)} = Z - alpha * V
        Y^{(i)} = Z - alpha/2 * W (Z + Y^{(i-1)})

    Shapes:
        Z : r x p
        W : r x r
        V : r x p
    """
    Y = Z - alpha * V
    for _ in range(s):
        Y = Z - 0.5 * alpha * (W @ (Z + Y))
    return Y

class SGDG(Optimizer):
    r"""Optimizer with two routines controlled by `stiefel`.

    If stiefel is True:
        generalized Stiefel update for row-shaped parameters Y satisfying
            Y B Y^T = I
        where B is provided manually in the parameter group.

    If stiefel is False:
        ordinary Euclidean SGD / momentum fallback.

    Args:
        params (iterable): iterable of parameters or dicts defining parameter groups

        Common parameters
        -----------------
        lr (float): learning rate
        momentum (float, optional): momentum factor for Euclidean branch and
            manifold first-moment accumulation (default: 0)
        stiefel (bool, optional): whether to use the generalized manifold update
            (default: False)

        Euclidean branch parameters
        ---------------------------
        weight_decay (float, optional): L2 penalty (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        Generalized Stiefel branch parameters
        -------------------------------------
        B (Tensor, required when stiefel=True): symmetric PSD matrix defining
            the constraint Y B Y^T = I
        grad_clip (float, optional): threshold for row-wise clipping of tangent
            gradient / momentum direction (default: None)
        project_every_step (bool, optional): re-project current parameter onto
            the manifold before each step (default: True)
        eps (float, optional): numerical threshold passed to retraction / helper
            routines (default: 1e-10)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 stiefel=False, B=None, grad_clip=None,
                 project_every_step=True, eps=1e-10,cayley_steps=2):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            B=B,
            grad_clip=grad_clip,
            project_every_step=project_every_step,
            eps=eps,
	    cayley_steps=cayley_steps,	
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(SGDG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('project_every_step', True)
            group.setdefault('eps', 1e-10)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            stiefel = group['stiefel']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            grad_clip = group['grad_clip']
            B = group['B']
            project_every_step = group['project_every_step']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                if stiefel:
    			cache = _get_B_cache(group, p.device, p.dtype)
    			C_t = cache["C_t"]              # r x n
    			C_dag_t = cache["C_dag_t"]      # n x r
   			P_range = cache["P_range"]      # n x n
    			r = cache["rank"]
    			s = group.get("cayley_steps", 2)

    			y = p.data.view(p.size(0), -1)       # p x n
    			g = p.grad.data.view(p.size(0), -1)  # p x n

    			if y.size(0) > r:
        			raise ValueError(
            			f"Need p <= rank(B). Got p={y.size(0)}, rank(B)={r}."
        			)

    			if weight_decay != 0:
        			g = g.add(y, alpha=weight_decay)

    			# Move to reduced ordinary-Stiefel coordinates
    			Z = _row_to_reduced(y, C_t)          # r x p
    			Gz = C_t @ g.transpose(0, 1)         # r x p

    			# Optional: project to support/range of B before reducing
    			# (mainly relevant if y has numerical nullspace drift)
    			X = y.transpose(0, 1)                # n x p
    			X = _project_to_range(X, P_range)
    			Z = C_t @ X

    			param_state = self.state[p]
    			if 'momentum_buffer' not in param_state:
        		param_state['momentum_buffer'] = torch.zeros_like(Z)   # r x p

    			V_prev = param_state['momentum_buffer']
    			V = momentum * V_prev - Gz

    			# Ordinary Stiefel Cayley-sequential update on Z
    			# This mirrors the original file, but now in reduced coordinates.
    			MX = torch.matmul(V, Z.transpose(0, 1))          # r x r
    			XMX = torch.matmul(Z.transpose(0, 1), MX)        # p x r
    			XXMX = torch.matmul(Z, XMX)                      # r x r
    			W_hat = MX - 0.5 * XXMX
    			W = W_hat - W_hat.transpose(0, 1)                # r x r skew

    			t = 1.0 / (matrix_norm_one(W) + EPS32)
    			alpha = min(lr, t)

    			Z_new = _sequential_cayley(Z, W, V, alpha, s=s)

    			# Map back only here
    			y_new = _reduced_to_row(Z_new, C_dag_t)

    			p.data.copy_(y_new.view_as(p.data))

   			# transport-like update for stored momentum:
    			# same spirit as original file: V_new = W @ Z_new
    			param_state['momentum_buffer'].copy_(V)
                else:
                    d_p = p.grad.data

                    if weight_decay != 0:
                        d_p = d_p.add(p.data, alpha=weight_decay)

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=(1.0 - dampening))

                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf

                    p.data.add_(d_p, alpha=-lr)

        return loss


class AdamG(Optimizer):
    r"""Adam-style optimizer with two routines controlled by `stiefel`.

    If stiefel is True:
        generalized Stiefel update for row-shaped parameters Y satisfying
            Y B Y^T = I
        where B is provided manually in the parameter group.

    If stiefel is False:
        Euclidean fallback using SGD-like momentum behavior, matching the
        spirit of the original code.

    Args:
        params (iterable): iterable of parameters or dicts defining parameter groups

        Common parameters
        -----------------
        lr (float): learning rate
        momentum (float, optional): used as beta1 in the manifold Adam branch,
            and as classical momentum in the Euclidean branch (default: 0)
        stiefel (bool, optional): whether to use generalized manifold Adam
            (default: False)

        Euclidean branch parameters
        ---------------------------
        weight_decay (float, optional): L2 penalty (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

        Generalized Stiefel branch parameters
        -------------------------------------
        B (Tensor, required when stiefel=True): symmetric PSD matrix defining
            the constraint Y B Y^T = I
        beta2 (float, optional): exponential decay rate for second moment
            (default: 0.99)
        epsilon (float, optional): numerical constant for stability
            (default: 1e-8)
        grad_clip (float, optional): threshold for row-wise clipping of tangent
            gradient / direction (default: None)
        project_every_step (bool, optional): re-project current parameter onto
            the manifold before each step (default: True)
        eps (float, optional): threshold passed to helper routines
            (default: 1e-10)
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 stiefel=False, B=None, beta2=0.99, epsilon=1e-8,
                 grad_clip=None, project_every_step=True, eps=1e-10, cayley_steps=2):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            B=B,
            beta2=beta2,
            epsilon=epsilon,
            grad_clip=grad_clip,
            project_every_step=project_every_step,
            eps=eps,
	    cayley_steps=cayley_steps,
        )

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super(AdamG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('project_every_step', True)
            group.setdefault('eps', 1e-10)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            stiefel = group['stiefel']
            lr = group['lr']
            beta1 = group['momentum']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            grad_clip = group['grad_clip']
            B = group['B']
            project_every_step = group['project_every_step']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                if stiefel:
    			cache = _get_B_cache(group, p.device, p.dtype)
    			C_t = cache["C_t"]              # r x n
    			C_dag_t = cache["C_dag_t"]      # n x r
    			P_range = cache["P_range"]      # n x n
    			r = cache["rank"]

    			y = p.data.view(p.size(0), -1)       # p x n
    			g = p.grad.data.view(p.size(0), -1)  # p x n

    			if y.size(0) > r:
        			raise ValueError(
            			f"Need p <= rank(B). Got p={y.size(0)}, rank(B)={r}."
        			)

    			# Optional ambient weight decay, same spirit as original fallback
    			if weight_decay != 0:
        		g = g.add(y, alpha=weight_decay)

    			# Keep parameter in/support of range(B)
    			X = y.transpose(0, 1)          # n x p
    			X = P_range @ X

    			# Reduced coordinates: ordinary Stiefel variable
    			Z = C_t @ X                    # r x p
    			Gz = C_t @ g.transpose(0, 1)   # r x p

    			param_state = self.state[p]
    			if 'm_buffer' not in param_state:
        			param_state['m_buffer'] = torch.zeros_like(Z)          # r x p
        			param_state['v_buffer'] = torch.zeros(
            			1, device=p.device, dtype=p.dtype
        			)
        			param_state['beta1_power'] = beta1
        			param_state['beta2_power'] = beta2

    			m = param_state['m_buffer']
    			v = param_state['v_buffer']
    			beta1_power = param_state['beta1_power']
    			beta2_power = param_state['beta2_power']

    			# Adam moments in reduced coordinates
    			mnew = beta1 * m + (1.0 - beta1) * Gz
    			vnew = beta2 * v + (1.0 - beta2) * (torch.norm(Gz) ** 2)

    			mhat = mnew / (1.0 - beta1_power)
    			vhat = vnew / (1.0 - beta2_power)

    			# Build skew matrix exactly in ordinary Stiefel reduced space
    			MX = torch.matmul(mhat, Z.transpose(0, 1))         # r x r
    			XMX = torch.matmul(Z.transpose(0, 1), MX)          # p x r
    			XXMX = torch.matmul(Z, XMX)                        # r x r
   			W_hat = MX - 0.5 * XXMX
    			W = (W_hat - W_hat.transpose(0, 1)) / torch.sqrt(vhat + epsilon)

    			t = 1.0 / (matrix_norm_one(W) + 1e-8)
    			alpha = min(lr, t)

    			# Sequential Cayley update on reduced variable
    			Z_new = Cayley_loop(Z, W, mnew, -alpha)

    			# Map back only here
    			y_new = _reduced_to_row(Z_new, C_dag_t)
    			p.data.copy_(y_new.view_as(p.data))

    			# Transport-like buffer update, matching the original Adam-Cayley spirit
    			mnew_transport = torch.matmul(W, Z) * torch.sqrt(vhat + epsilon) * (1.0 - beta1_power)

    			m.copy_(mnew_transport)
    			v.copy_(vnew)

    			param_state['beta1_power'] *= beta1
    			param_state['beta2_power'] *= beta2

                else:
                    d_p = p.grad.data

                    if weight_decay != 0:
                        d_p = d_p.add(p.data, alpha=weight_decay)

                    if beta1 != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = d_p.clone()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(beta1).add_(d_p, alpha=(1.0 - dampening))

                        if nesterov:
                            d_p = d_p.add(buf, alpha=beta1)
                        else:
                            d_p = buf

                    p.data.add_(d_p, alpha=-lr)

        return loss
