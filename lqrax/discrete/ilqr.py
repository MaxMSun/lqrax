import numpy as np 
from functools import partial

import jax 
import jax.numpy as jnp 
from jax import jit, vmap, grad, jacfwd 
from jax.lax import scan 


class iLQR:
    def __init__(self, 
                 x_dim: int, 
                 u_dim: int, 
                 Q: None, 
                 R: None) -> None: 
        self.x_dim = x_dim 
        self.u_dim = u_dim 

        self.Q = Q 
        self.Q_inv = jnp.linalg.inv(Q) if self.Q is not None else None
        self.R = R 
        self.R_inv = jnp.linalg.inv(R) if self.R is not None else None

        self._dfdx = jacfwd(self.dyn, argnums=0)
        self._dfdu = jacfwd(self.dyn, argnums=1)

    def dyn(self, xt, ut):
        raise NotImplementedError("Dynamics function f(xt, ut) not implemented.")

    def dyn_step(self, xt, ut):
        xt_new = self.dyn(xt, ut)
        return xt_new, xt_new 

    @partial(jit, static_argnums=(0,))
    def traj_sim(self, x0, u_traj):
        xT, x_traj = scan(f=self.dyn_step, init=x0, xs=u_traj)
        return x_traj

    def dfdx(self, xt, ut):
        return self._dfdx(xt, ut)

    def dfdu(self, xt, ut):
        return self._dfdu(xt, ut)

    @partial(jit, static_argnums=(0,))
    def linearize_dyn(self, x0, u_traj):
        x_traj = self.traj_sim(x0, u_traj)
        A_traj = vmap(self.dfdx, in_axes=(0,0))(x_traj, u_traj)
        B_traj = vmap(self.dfdu, in_axes=(0,0))(x_traj, u_traj)
        return x_traj, A_traj, B_traj

    def P_dyn_rev_step(self, Pt, At_Bt):
        At, Bt = At_Bt
        Pt_new = self.Q + At.T @ Pt @ At - At.T @ Pt @ Bt @ jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt) @ Bt.T @ Pt @ At
        return Pt_new, Pt_new

    @partial(jit, static_argnums=(0,))
    def P_dyn_rev_scan(self, PT, A_traj, B_traj):
        _, P_traj_rev = scan(
            f=self.P_dyn_rev_step,
            init=PT,
            xs=(A_traj, B_traj)
        )
        return P_traj_rev

    def r_dyn_rev_step(self, rt, Pt_At_Bt_at_bt):
        Pt, At, Bt, at, bt = Pt_At_Bt_at_bt
        Kt = jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt) @ Bt.T @ Pt @ At
        kt = jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt) @ (Bt.T @ (Pt @ at + rt) + bt)
        rt_new = (At - Bt @ Kt).T @ rt + at - Bt @ kt
        return rt_new, rt_new

    @partial(jit, static_argnums=(0,))
    def r_dyn_rev_scan(self, rT, P_traj, A_traj, B_traj, a_traj, b_traj):
        _, r_traj_rev = scan(
            f=self.r_dyn_rev_step, 
            init=rT,
            xs=(P_traj, A_traj, B_traj, a_traj, b_traj)
        )
        return r_traj_rev

    def z2v(self, zt, Pt, rt, Bt, bt):
        Ginv = jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt)
        Kt = Ginv @ Bt.T @ Pt
        kt = Ginv @ (Bt.T @ rt + bt)
        return -Kt @ zt - kt

    def z_dyn_step(self, zt, Pt_rt_At_Bt_bt):
        Pt, rt, At, Bt, bt = Pt_rt_At_Bt_bt
        vt = self.z2v(zt, Pt, rt, Bt, bt)
        zt_new = At @ zt + Bt @ vt
        return zt_new, zt_new

    @partial(jit, static_argnums=(0,))
    def z_traj_sim(self, z0, P_traj, r_traj, A_traj, B_traj, b_traj):
        _, z_traj = scan(
            f=self.z_dyn_step,
            init=z0,
            xs=(P_traj, r_traj, A_traj, B_traj, b_traj)
        )
        return z_traj

    @partial(jit, static_argnums=(0,))
    def solve(self, A_traj, B_traj, a_traj, b_traj):
        A_traj_rev = A_traj[::-1]
        B_traj_rev = B_traj[::-1]
        a_traj_rev = a_traj[::-1]
        b_traj_rev = b_traj[::-1]

        PT = jnp.zeros((self.x_dim, self.x_dim))
        P_traj_rev = self.P_dyn_rev_scan(PT, A_traj_rev, B_traj_rev)
        P_traj = P_traj_rev[::-1]

        rT = jnp.zeros(self.x_dim)
        r_traj_rev = self.r_dyn_rev_scan(rT, P_traj_rev, A_traj_rev, B_traj_rev, a_traj_rev, b_traj_rev)
        r_traj = r_traj_rev[::-1]

        z0 = jnp.zeros(self.x_dim)
        z_traj = self.z_traj_sim(z0, P_traj, r_traj, A_traj, B_traj, b_traj)
        v_traj = vmap(self.z2v, in_axes=(0,0,0,0,0))(z_traj, P_traj, r_traj, B_traj, b_traj)

        return v_traj, z_traj