import numpy as np 
from functools import partial

import jax 
import jax.numpy as jnp 
from jax import jit, vmap, grad, jacfwd 
from jax.lax import scan 


class iLQR:
    def __init__(self, 
                 dt: float, 
                 x_dim: int, 
                 u_dim: int, 
                 Q: None, 
                 R: None) -> None: 
        self.dt = dt  
        self.x_dim = x_dim 
        self.u_dim = u_dim 

        self.Q = Q 
        self.Q_inv = jnp.linalg.inv(Q) if self.Q != None else None
        self.R = R 
        self.R_inv = jnp.linalg.inv(R) if self.R != None else None

        self._dfdx = jacfwd(self.dyn, argnums=0)
        self._dfdu = jacfwd(self.dyn, argnums=1)
        self._dldx = grad(self.loss, argnums=0)
        self._dldu = grad(self.loss, argnums=1)

    def dyn(self, xt, ut):
        raise NotImplementedError("Dynamics function f(xt, ut) not implemented.")
    
    def loss(self, xt, ut):
        raise NotImplementedError("Dynamics function f(xt, ut) not implemented.")
    
    def dyn_step(self, xt, ut):
        k1 = self.dt * self.dyn(xt, ut)
        k2 = self.dt * self.dyn(xt + k1/2.0, ut)
        k3 = self.dt * self.dyn(xt + k2/2.0, ut)
        k4 = self.dt * self.dyn(xt + k3, ut)
        xt_new = xt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0
        return xt_new, xt_new 
    
    @partial(jit, static_argnums=(0,))
    def dyn_scan(self, x0, u_traj):
        xT, x_traj = scan(f=self.dyn_step, init=x0, xs=u_traj)
        return x_traj
    
    def dfdx(self, xt, ut):
        return self._dfdx(xt, ut)
    
    def dfdu(self, xt, ut):
        return self._dfdu(xt, ut)
    
    def dldx(self, xt, ut):
        return self._dldx(xt, ut)
    
    def dldu(self, xt, ut):
        return self._dldu(xt, ut)
    
    @partial(jit, static_argnums=(0,))
    def linearize_dyn(self, x0, u_traj):
        x_traj = self.dyn_scan(x0, u_traj)
        A_traj = vmap(self.dfdx, in_axes=(0,0))(x_traj, u_traj)
        B_traj = vmap(self.dfdu, in_axes=(0,0))(x_traj, u_traj)
        return A_traj, B_traj
    
    @partial(jit, static_argnums=(0,))
    def linearize_loss(self, x0, u_traj):
        x_traj = self.dyn_scan(x0, u_traj)
        a_traj = vmap(self.dldx, in_axes=(0,0))(x_traj, u_traj)
        b_traj = vmap(self.dldu, in_axes=(0,0))(x_traj, u_traj)
        return a_traj, b_traj

    def P_dyn_rev(self, Pt, At, Bt):
        return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_inv @ Bt.T @ Pt + self.Q 

    def P_dyn_rev_step(self, Pt, At_Bt):
        At, Bt = At_Bt
        k1 = self.dt * self.P_dyn_rev(Pt, At, Bt)
        k2 = self.dt * self.P_dyn_rev(Pt+k1/2, At, Bt)
        k3 = self.dt * self.P_dyn_rev(Pt+k2/2, At, Bt)
        k4 = self.dt * self.P_dyn_rev(Pt+k3, At, Bt)

        Pt_new = Pt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return Pt_new, Pt_new 
    
    @partial(jit, static_argnums=(0,))
    def P_dyn_rev_scan(self, PT, A_traj, B_traj):
        P0, P_traj_rev = scan(
            f=self.P_dyn_rev_step,
            init=PT,
            xs=(A_traj, B_traj)
        )
        return P_traj_rev
    
    def r_dyn_rev(self, rt, Pt, At, Bt, at, bt):
        return (At - Bt @ self.R_inv @ Bt.T @ Pt).T @ rt + at - Pt @ Bt @ self.R_inv @ bt

    def r_dyn_rev_step(self, rt, Pt_At_Bt_at_bt):
        Pt, At, Bt, at, bt = Pt_At_Bt_at_bt
        k1 = self.dt * self.r_dyn_rev(rt, Pt, At, Bt, at, bt)
        k2 = self.dt * self.r_dyn_rev(rt+k1/2, Pt, At, Bt, at, bt)
        k3 = self.dt * self.r_dyn_rev(rt+k2/2, Pt, At, Bt, at, bt)
        k4 = self.dt * self.r_dyn_rev(rt+k3, Pt, At, Bt, at, bt)

        rt_new = rt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return rt_new, rt_new

    @partial(jit, static_argnums=(0,))
    def r_dyn_rev_scan(self, rT, P_traj, A_traj, B_traj, a_traj, b_traj):
        r0, r_traj_rev = scan(
            f=self.r_dyn_rev_step, 
            init=rT,
            xs=(P_traj, A_traj, B_traj, a_traj, b_traj)
        )
        return r_traj_rev

    def z2v(self, zt, Pt, rt, Bt, bt):
        return -self.R_inv @ Bt.T @ Pt @ zt - self.R_inv @ Bt.T @ rt - self.R_inv @ bt

    def z_dyn(self, zt, Pt, rt, At, Bt, bt):
        return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)
        
    def z_dyn_step(self, zt, Pt_rt_At_Bt_bt):
        Pt, rt, At, Bt, bt = Pt_rt_At_Bt_bt
        k1 = self.dt * self.z_dyn(zt, Pt, rt, At, Bt, bt)
        k2 = self.dt * self.z_dyn(zt+k1/2, Pt, rt, At, Bt, bt)
        k3 = self.dt * self.z_dyn(zt+k2/2, Pt, rt, At, Bt, bt)
        k4 = self.dt * self.z_dyn(zt+k3, Pt, rt, At, Bt, bt)

        zt_new = zt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return zt_new, zt_new
    
    @partial(jit, static_argnums=(0,))
    def z_dyn_scan(self, z0, P_traj, r_traj, A_traj, B_traj, b_traj):
        zT, z_traj = scan(
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
        z_traj = self.z_dyn_scan(z0, P_traj, r_traj, A_traj, B_traj, b_traj)
        v_traj = vmap(self.z2v, in_axes=(0,0,0,0,0))(z_traj, P_traj, r_traj, B_traj, b_traj)

        return v_traj