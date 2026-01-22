import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
from model.dynamics import SingleIntegrator
import torch.distributions as td
from model.gmm3d import GMM3D
from model.gmm2d import GMM2D

class MobileMPPI:
    def __init__(self, horizon, dim, gamma, device, dt):
        self.horizon = horizon
        self.action_dim = dim
        self.device = device

        # reward param
        self.gamma = [gamma**h for h in range(horizon)]
        self.gamma = torch.tensor(self.gamma, dtype=torch.float32).to(self.device)

        # dynamics
        self.dt = dt
        self.T = horizon
        self.dynamic = SingleIntegrator(self.dt, {}, device, 2)

        # col cost
        self.rho_0 = 1.5
        self.rho_1 = 3
        self.eta = 3

        # goal cost
        self.nu = 1
        self.log_sigma = np.log(self.nu**2)
        self.prob_grid = None

        # intent cost
        self.kappa = 1
        self.modes = 10
        

        # mppi param
        self.N = 1000 # sample number
        self.param_lambda = 2
        self.nominal_u = torch.zeros(1, 1, self.T, self.action_dim).to(self.device)
        self.opt_iters = 1
        self.alpha_mu = 0.2
        self.alpha_cov = 0.2
        
    
    def col_cost(self, states, obstacles):
        """
        states: tensor(N, 1, h, dim)
        obstacles: tensor(n, dim)
        """
        N, _, h, dim = states.shape
        obstacles = obstacles[...,:dim]
        col_cost = torch.zeros_like(states[:,0,0,0])
    
        rho = torch.norm(states - obstacles[None, :, None, :], p=2, dim=-1) # (N, n, h)
        rho = rho.min(-1)[0] # (N, n)
        rho = rho.min(-1)[0] # (N,)
        col_cost[rho<self.rho_1] = 30*((1/rho-self.rho_0) - (1/self.rho_1-self.rho_0))[rho<self.rho_1]

        col_cost[rho<self.rho_0] = 9999
        return col_cost

    
    def goal_cost(self, states, goals):
        """
        states: tensor(N, 1, h, dim)
        goals: tensor(n, dim)
        """
        n_goals, _ = goals.shape
        N, _, h, dim = states.shape
        
        if dim==2:
            gmm_class = GMM2D
        else:
            gmm_class = GMM3D
        goals = goals[...,:dim]
        goal_gmm = gmm_class(log_pis=torch.zeros_like(goals[:,0].reshape(1,1,1,n_goals)), mus=goals.reshape(1,1,1,n_goals,dim), log_sigmas=self.log_sigma*torch.ones_like(goals.reshape(1,1,1,n_goals,dim)), corrs=torch.zeros_like(goals[:,0].reshape(1,1,1,n_goals)))
        goal_cost = -goal_gmm.log_prob(states.reshape(N, 1, h, dim)[...,:,:]).mean(1).min(-1)[0] 
        return goal_cost, 0.5
    
    def intent_cost(self, states):
        """
        states: tensor(N, 1, h, 3) or (N, 1, h, 2)
        v_u: tensor(3,) or (2)
        u0: tensor(N, 3) or (N, 2)
        """
        pgmm = self.dist
        vgmm = self.v_dist
        N, _, h, dim = states.shape
        assert h == self.horizon
        
        neg_log_p_y_xr = - pgmm.log_prob(states.reshape(N, 1, h, dim)).mean(1) 
        neg_log_p_y_xr = neg_log_p_y_xr * self.gamma.reshape(1, h)

        intent_cost = neg_log_p_y_xr.mean(-1)

        return intent_cost
    
    def compute_weight(self, S):
        """compute weights for each sample"""
        # prepare buffer
        # w = torch.zeros((self.N)).to(self.device)

        # calculate rho
        rho = S.min()

        # calculate eta
        eta = torch.exp( (-1.0/self.param_lambda) * (S-rho) ).sum()

        # calculate weight
        w = (1.0 / eta) * torch.exp( (-1.0/self.param_lambda) * (S-rho) )
        return w
    
    def average_filter(self, u, window_size=3):
        """
        u: tensor(H, dim)
        """
        kernel = (torch.ones((2, 1, window_size)) / window_size).to(self.device)
        padding = (window_size - 1)//2
        u_padded = F.pad(u.transpose(1,0), (padding, padding), mode='replicate')

        u_filtered = F.conv1d(u_padded.unsqueeze(0), kernel, groups=self.action_dim)
        return u_filtered.squeeze(0).transpose(1,0)
    
    def cost_computation(self, x_perturb, obstacles):
        cost = 0
        c_i = self.intent_cost(x_perturb)
        
        # c_g, w_g = self.goal_cost(x_perturb, goals)
        # cost = c_g * (w_g/ self.nu) + c_i * (1 - w_g/ self.nu)
        cost = c_i#* (w_g) + c_i * (1 - w_g)
        if obstacles.shape[0] > 0:
            cost += self.col_cost(x_perturb, obstacles)
        return cost
    

    @torch.no_grad()
    def plan_action(self, cur_state, dist, v_dist, goals, obstacles):
        """
        cur_state: tensor(3,) or (2,)
        goals: tensor(K, 3) or (K, 3)
        obstacles: tensor(N, 3) or (N, 3)
        """
        t = time.time()
        self.dist = dist
        self.v_dist = v_dist

        obstacles = torch.tensor(obstacles, dtype=torch.float32).to(self.device)
        goals = torch.tensor(goals, dtype=torch.float32).to(self.device)

        cur_state = torch.tensor(cur_state, dtype=torch.float32).to(self.device)
        self.dynamic.set_initial_condition({'pos':cur_state.reshape(-1, 1, self.action_dim).repeat(self.N, 1, 1)})

        mus = self.v_dist.mus
        cov = self.v_dist.cov

        for opt_iter in range(self.opt_iters):
            sample_dist = GMM2D.from_log_pis_mus_cov_mats(torch.softmax(self.v_dist.log_pis*0.0, dim=-1).log(), mus, cov)
            u_perturb, modes = sample_dist.rsample(sample_shape=torch.Size([self.N]), output_mode=True)
            u_perturb = u_perturb.reshape(self.N, 1, self.T, self.action_dim) # (N, 1, T, dim)
            # u_perturb_extra = torch.randn(size=torch.Size([100*self.T, self.action_dim])).reshape(100, 1, self.T, self.action_dim).to(self.device) *0.1 + self.nominal_u
            # u_perturb = torch.cat((u_perturb, u_perturb_extra),dim=0)
            x_perturb = self.dynamic.integrate_samples(u_perturb) # (N, 1, T, dim)

            # cost calculation
            cost = self.cost_computation(x_perturb, obstacles)
            # v, indices = torch.topk(-cost, 10)

            # action cost
            action_cost = ((u_perturb-self.nominal_u)**2).sum((1,2,3)) *0.5
            cost += action_cost

            # compute best u for each mode
            us = []
            covs = []
            for mode in range(10):
                mask = modes[...,mode,0] == 1
                cost_mode = cost[mask.reshape(-1)]
                w_mode = self.compute_weight(cost_mode)
                us.append((u_perturb[mask.reshape(-1)] * w_mode.reshape(-1,1,1,1)).sum(0))
                diff = (u_perturb[mask.reshape(-1)] - mus[...,mode,:]).unsqueeze(-1)
                covs.append( (torch.bmm(diff.reshape(-1,2,1), diff.reshape(-1,2,1).permute(0,2,1)).reshape(diff.shape[:3]+(2,2))* w_mode.reshape(-1,1,1,1,1)).sum(0)  )
            mus = mus * (1-self.alpha_mu) + self.alpha_mu * torch.stack(us, dim=-2).unsqueeze(0)
            cov = cov * (1-self.alpha_cov) + self.alpha_cov * torch.stack(covs, dim=-3).unsqueeze(0)

    
        w_all = self.compute_weight(cost)
        us.append((u_perturb*w_all.reshape(-1,1,1,1)).sum(0))

    
        us = torch.stack(us, dim=0)
        
        self.dynamic.set_initial_condition({'pos':cur_state.reshape(-1, 1, self.action_dim).repeat(us.shape[0], 1, 1)})
        xs = self.dynamic.integrate_samples(us.reshape(us.shape[0],1,-1,self.action_dim))

        cost = self.cost_computation(xs, obstacles)
        best_mode = torch.argmin(cost)

        u = us[best_mode][0]

        u = self.average_filter(u)
        self.nominal_u[...,:-1,:] = u.reshape(1,1,-1,self.action_dim)[...,1:,:]
        self.nominal_u[...,-1,:] = u.reshape(1,1,-1,self.action_dim)[...,0,:]

        low_cost_traj = []
        for i in range(10):
            low_cost_traj.append(xs[i].cpu().numpy())
        low_cost_traj.append(xs[best_mode].cpu().numpy())

        return u[0], low_cost_traj
        # return v_u

        








        


