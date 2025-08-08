import numpy as np 
from functools import partial

import jax 
import jax.numpy as jnp 
from jax import jit, vmap, grad, jacfwd 
from jax.lax import scan 


class LQR:
    def __init__(self, 
                 x_dim: int, 
                 u_dim: int, 
                 Q: jnp.ndarray, 
                 R: jnp.ndarray) -> None:
        self.x_dim = x_dim
        self.u_dim = u_dim

        self.Q = Q
        self.Q_inv = jnp.linalg.inv(Q)
        self.R = R
        self.R_inv = jnp.linalg.inv(R)

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
        A_traj = vmap(self.dfdx)(x_traj, u_traj)
        B_traj = vmap(self.dfdu)(x_traj, u_traj)
        return x_traj, A_traj, B_traj

    def P_dyn_rev(self, Pt, At, Bt):
        return At.T @ Pt @ At - At.T @ Pt @ Bt @ jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt) @ Bt.T @ Pt @ At + self.Q

    def P_dyn_rev_step(self, Pt, At_Bt):
        At, Bt = At_Bt
        Pt_new = self.P_dyn_rev(Pt, At, Bt)
        return Pt_new, Pt_new

    @partial(jit, static_argnums=(0,))
    def P_dyn_rev_scan(self, PT, A_traj, B_traj):
        P0, P_traj_rev = scan(
            f=self.P_dyn_rev_step,
            init=PT,
            xs=(A_traj, B_traj)
        )
        return P_traj_rev

    def r_dyn_rev(self, rt, Pt, At, Bt, at):
        Kt = jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt) @ Bt.T @ Pt @ At
        return (At - Bt @ Kt).T @ rt - self.Q @ at

    def r_dyn_rev_step(self, rt, Pt_At_Bt_at):
        Pt, At, Bt, at = Pt_At_Bt_at
        rt_new = self.r_dyn_rev(rt, Pt, At, Bt, at)
        return rt_new, rt_new

    @partial(jit, static_argnums=(0,))
    def r_dyn_rev_scan(self, rT, P_traj, A_traj, B_traj, a_traj):
        r0, r_traj_rev = scan(
            f=self.r_dyn_rev_step,
            init=rT,
            xs=(P_traj, A_traj, B_traj, a_traj)
        )
        return r_traj_rev

    @partial(jit, static_argnums=(0,))
    def z2v(self, zt, Pt, rt, Bt):
        Kt = jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt) @ Bt.T @ Pt
        kt = jnp.linalg.inv(self.R + Bt.T @ Pt @ Bt) @ Bt.T @ rt
        return -Kt @ zt - kt

    def z_dyn(self, zt, Pt, rt, At, Bt):
        return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt)

    def z_dyn_step(self, zt, Pt_rt_At_Bt):
        Pt, rt, At, Bt = Pt_rt_At_Bt
        zt_new = self.z_dyn(zt, Pt, rt, At, Bt)
        return zt_new, zt_new

    @partial(jit, static_argnums=(0,))
    def z_traj_sim(self, z0, P_traj, r_traj, A_traj, B_traj):
        zT, z_traj = scan(
            f=self.z_dyn_step,
            init=z0,
            xs=(P_traj, r_traj, A_traj, B_traj)
        )
        return z_traj

    @partial(jit, static_argnums=(0,))
    def solve(self, x0, A_traj, B_traj, ref_traj):
        A_traj_rev = A_traj[::-1]
        B_traj_rev = B_traj[::-1]
        a_traj_rev = ref_traj[::-1]

        PT = jnp.zeros((self.x_dim, self.x_dim))
        P_traj_rev = self.P_dyn_rev_scan(PT, A_traj_rev, B_traj_rev)
        P_traj = P_traj_rev[::-1]

        rT = jnp.zeros(self.x_dim)
        r_traj_rev = self.r_dyn_rev_scan(rT, P_traj_rev, A_traj_rev, B_traj_rev, a_traj_rev)
        r_traj = r_traj_rev[::-1]

        z0 = x0
        z_traj = self.z_traj_sim(z0, P_traj, r_traj, A_traj, B_traj)
        v_traj = vmap(self.z2v)(z_traj, P_traj, r_traj, B_traj)

        return v_traj, z_traj
