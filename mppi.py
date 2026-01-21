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
    def __init__(self, horizon, dim, gamma, device, XY, dt):
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
        self.XY = XY
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
        

        self.u = torch.autograd.Variable(torch.zeros((self.horizon, self.action_dim))).to(self.device)
        self.u.requires_grad = True
        self.optimizer = optim.Adam([self.u], lr=0.5)
        self.escape=False
    

    # def col_cost(self, states, obstacles):
    #     """
    #     states: tensor(N, 1, h, dim)
    #     obstacles: tensor(n, dim)
    #     """
    #     # rho = torch.norm(states - obstacles[None, :, None, :], p=2, dim=-1) # (N, n, h)
    #     # mask = rho <= self.rho_0
    #     # col_cost = self.eta * (1/torch.abs(rho - self.rho_0)) * self.gamma.reshape(1,1,-1)
    #     # col_cost = col_cost[mask].sum()
    #     # return col_cost
    #     N, _, h, dim = states.shape
    #     obstacles = obstacles[...,:dim]
    
    #     rho = torch.norm(states - obstacles[None, :, None, :], p=2, dim=-1) # (N, n, h)
    #     rho[rho<0.0001] = 0.0001
    #     # mask = rho <= self.rho_0
    #     col_cost = (self.eta * torch.exp(- self.eta * (1/rho))).min(1)[0] # (N, h)
    #     # * self.gamma.reshape(1,1,-1)
    #     col_cost = - col_cost.log() * self.gamma.reshape(1,-1)
    #     col_cost = col_cost.sum(-1)
    #     return col_cost
    
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
    
    # def goal_cost(self, states, goals):
    #     """
    #     states: tensor(N, 1, h, dim)
    #     goals: tensor(n, dim)
    #     """
    #     pgmm = self.dist
    #     n_goals = goals.shape[0]
    #     _, _, t, n_modes, dim = pgmm.mus.shape
    #     # assert self.horizon == h
    #     goal_cost = torch.norm(states- goals[None, :,None], p=2, dim=-1) # (N, n_goals, h)
    #     # weight = self.nu * torch.exp(pgmm.log_prob(goals.reshape(1, n_goals, 1, self.action_dim))) # (1, n_goals, timesteps)
    #     weight = self.nu * torch.softmax(torch.exp(pgmm.log_prob(goals.reshape(1, n_goals, 1, self.action_dim))).max(-1)[0], dim=-1) # (1, n_goals, timesteps)
    #     goal_cost = (weight.reshape(1, n_goals, 1) * goal_cost.reshape(-1, n_goals, self.horizon) * self.gamma.reshape(1, 1, self.horizon)).sum((-2,-1))
    #     return goal_cost, weight.max(-1)[0]
    
    # def goal_cost(self, states, goals):
    #     """
    #     states: tensor(N, 1, h, dim)
    #     goals: tensor(n, dim)
    #     """
    #     n_goals = goals.shape[0]
    #     search_grid = torch.stack(self.XY, dim=2).view(-1, 2).float().to(self.device)
    #     # score = torch.exp(nt_gmm.log_prob(search_grid))#/torch.exp(nt_gmm.log_prob(nt_gmm.mus.reshape(-1,3))).max()

    #     # prob_grid = torch.mean(score, dim=0).reshape(score[0].shape)
    #     # prob_grid = prob_grid/prob_grid.sum()

    #     x_mask = (search_grid[:,None, 0]-goals[None,:, 0]).abs() <= 15*0.08
    #     y_mask = (search_grid[:,None, 1]-goals[None,:, 1]).abs() <= 15*0.08

    #     prob = []
    #     search_grid.unsqueeze(1).repeat(1,n_goals,1)[x_mask*y_mask, :]
    #     for i in range(n_goals):
    #         dist = ((search_grid[x_mask[:,i]*y_mask[:,i], :] - goals[i:i+1,:])**2).sum(-1).sqrt()
    #         dist = 1.0/(dist+1e-6)
    #         w_blinear = dist/dist.sum()
    #         prob.append((self.prob_grid.reshape(-1)[x_mask[:,i]*y_mask[:,i]]*w_blinear).sum())
        
    #     weight = torch.stack(prob, dim=0) * self.nu
    #     # weight = weight/(weight.sum()+1e-5)
    #     goal_cost = torch.norm(states - goals[None, :,None], p=2, dim=-1) # (N, n_goals, h)
    #     goal_cost = (weight.reshape(1, n_goals, 1) * goal_cost.reshape(-1, n_goals, self.horizon) * self.gamma.reshape(1, 1, self.horizon)).sum((-2,-1))
    #     return goal_cost, min(0.9, weight.max(-1)[0])
    
    # def goal_cost(self, states, goals):
    #     """
    #     states: tensor(N, 1, h, dim)
    #     goals: tensor(n, dim)
    #     """
    #     n_goals = goals.shape[0]
        
    #     goal_cost = -torch.norm(states - goals[None, :,None], p=2, dim=-1)[:,:,:] # (N, n_goals, h)
    #     goal_cost = -torch.log(((1/self.nu)*goal_cost.exp()).mean((1,2)))
    #     # goal_cost = (weight.reshape(1, n_goals, 1) * goal_cost.reshape(-1, n_goals, self.horizon) * self.gamma.reshape(1, 1, self.horizon)).sum((-2,-1))
    #     return goal_cost, 0.5
    
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
    
    def intent_cost(self, states, v_u, u0):
        """
        states: tensor(N, 1, h, 3) or (N, 1, h, 2)
        v_u: tensor(3,) or (2)
        u0: tensor(N, 3) or (N, 2)
        """
        pgmm = self.dist
        vgmm = self.v_dist
        N, _, h, dim = states.shape
        assert h == self.horizon
        
        n_modes = vgmm.mus.shape[-2]

        if v_u is not None:
            v_norm = torch.norm(v_u)
            p_r_vu = td.MultivariateNormal(v_u.reshape(-1, self.action_dim), torch.eye(self.action_dim).to(self.device)*1)
            neg_log_p_r_vu = - p_r_vu.log_prob(u0)


        # log_p_r_x = torch.exp(pgmm.log_pis)
        # log_p_r_x = log_p_r_x.reshape(log_p_r_x.shape[-1])

        # neg_log_p_y_xr = - pgmm.log_prob(states.reshape(N, h, 1, dim)).mean(-1)  # (N, h, t)
        pis = pgmm.log_pis.exp()[0,0,0,:]
        v_mode = vgmm.mus[0,0,0,:,:]
        # v_sigma = vgmm.log_sigmas[0,0,0,:,:]
        p_r_xcvu = td.MultivariateNormal(v_mode.reshape(-1, self.action_dim), vgmm.cov[0,0,0])

        if v_u is not None:
            pis_lh = p_r_xcvu.log_prob(v_u.reshape(-1,2).repeat(n_modes,1)).exp()

            # pis_lh = p_r_vu.log_prob(v_mode).exp()
            pis_posterior = pis*(pis_lh + 1e-5)
            pis_posterior = pis_posterior/pis_posterior.sum()

            # pgmm.pis_cat_dist = td.Categorical(logits=pis_posterior.log())
            # pgmm.log_pis = pis_posterior.log().reshape(1,1,1,-1).repeat(1,1,h,1)

        neg_log_p_y_xr = - pgmm.log_prob(states.reshape(N, 1, h, dim)).mean(1) 
        neg_log_p_y_xr = neg_log_p_y_xr * self.gamma.reshape(1, h)

        # neg_log_p_y_xr = []
        # for t in range(h):
        #     t_dist = pgmm.get_at_time(t)
        #     log_score = -t_dist.log_prob(states.reshape(1, h, 1, dim)[:,t:t+1]).reshape(-1)
        #     neg_log_p_y_xr.append(log_score)
        # neg_log_p_y_xr = torch.stack(neg_log_p_y_xr)
        # neg_log_p_y_xr = neg_log_p_y_xr * self.gamma.reshape(h,1)
        
        intent_cost = neg_log_p_y_xr.mean(-1) #+ neg_log_p_r_vu 

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
    
    def cost_computation(self, x_perturb, v_u, u_perturb, goals, obstacles):
        cost = 0
        c_i = self.intent_cost(x_perturb, v_u, u_perturb[:,0,0,:])
        
        # c_g, w_g = self.goal_cost(x_perturb, goals)
        # cost = c_g * (w_g/ self.nu) + c_i * (1 - w_g/ self.nu)
        cost = c_i#* (w_g) + c_i * (1 - w_g)
        if obstacles.shape[0] > 0:
            cost += self.col_cost(x_perturb, obstacles)
        return cost
    
    def escape_velo(self, cur_state, obstacles):
        u_escape1 = []
        u_escape2 = []
        distance = torch.norm(cur_state[None]-obstacles,p=2,dim=1) # (N)
        min_obs_index = torch.argmin(distance)
        cloest_obs = obstacles[min_obs_index]
        
        # escape velocity
        pos2obs = cloest_obs - cur_state
        escape_vec1 = torch.stack((-pos2obs[1], pos2obs[0]),dim=0)
        escape_vec1 = escape_vec1/torch.norm(escape_vec1)#*v_u_norm
        escape_vec2 = -escape_vec1
        u_escape1.append(escape_vec1)
        u_escape2.append(escape_vec2)

        imagine_state = cur_state + escape_vec1 * self.dt
        for t in range(1, self.T):
            distance = torch.norm(imagine_state[None]-obstacles,p=2,dim=1) # (N)
            min_obs_index = torch.argmin(distance)
            cloest_obs = obstacles[min_obs_index]
            
            # escape velocity
            pos2obs = cloest_obs - cur_state
            escape_vec = torch.stack((-pos2obs[1], pos2obs[0]),dim=0)
            escape_vec = escape_vec/torch.norm(escape_vec)#*v_u_norm
            if (escape_vec*u_escape1[-1]).sum()> 0:
                u_escape1.append(escape_vec)
            else:
                u_escape1.append(-escape_vec)
            imagine_state = imagine_state + escape_vec * self.dt
        
        imagine_state = cur_state + escape_vec2 * self.dt
        for t in range(1, self.T):
            distance = torch.norm(imagine_state[None]-obstacles,p=2,dim=1) # (N)
            min_obs_index = torch.argmin(distance)
            cloest_obs = obstacles[min_obs_index]
            
            # escape velocity
            pos2obs = cloest_obs - cur_state
            escape_vec = torch.stack((-pos2obs[1], pos2obs[0]),dim=0)
            escape_vec = escape_vec/torch.norm(escape_vec)#*v_u_norm
            if (escape_vec*u_escape2[-1]).sum()> 0:
                u_escape2.append(escape_vec)
            else:
                u_escape2.append(-escape_vec)
            imagine_state = imagine_state + escape_vec * self.dt
        u_escape1 = torch.stack(u_escape1, dim=0).view(1, 1, self.T, self.action_dim)
        u_escape2 = torch.stack(u_escape2, dim=0).view(1, 1, self.T, self.action_dim)
            
        return u_escape1, u_escape2

    def plan_action(self, cur_state, v_u, dist, v_dist, goals, obstacles):
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
        if v_u is not None:
            v_u = torch.tensor(v_u, dtype=torch.float32).to(self.device)
            v_u_norm = torch.norm(v_u)

        cur_state = torch.tensor(cur_state, dtype=torch.float32).to(self.device)
        self.dynamic.set_initial_condition({'pos':cur_state.reshape(-1, 1, self.action_dim).repeat(self.N, 1, 1)})
        
        # escape trajectory
        if obstacles.shape[0] > 0 and self.escape==True:
            u_escape1 = []
            u_escape2 = []
            # distance = torch.norm(cur_state[None]-obstacles,p=2,dim=1) # (N)
            # min_obs_index = torch.argmin(distance)
            # cloest_obs = obstacles[min_obs_index]
            
            # # escape velocity
            # pos2obs = cloest_obs - cur_state
            # escape_vec1 = torch.stack((-pos2obs[1], pos2obs[0]),dim=0)
            # escape_vec1 = escape_vec1/torch.norm(escape_vec1)*3
            # u_escape1 = escape_vec1.view(1, 1, 1, self.action_dim).repeat(1,1,self.T,1)        
            # gamma_u = [0.7**t for t in range(self.T)]
            # gamma_u = torch.tensor(self.gamma, dtype=torch.float32).view(1, 1, self.T, 1).to(self.device)
            # u_escape1 = u_escape1 * gamma_u

            # u_escape2 = -u_escape1
            u_escape1, u_escape2 = self.escape_velo(cur_state, obstacles)



        # nominal_trajectory = self.v_dist.mus # (1, 1, T, components, dim)
        log_pis = self.v_dist.log_pis.clone()
        # pis_cat_dist = td.Categorical(logits=log_pis)
        # self.v_dist.pis_cat_dist = td.Categorical(logits=torch.softmax(self.v_dist.log_pis*0.0, dim=-1).log())
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
            cost = self.cost_computation(x_perturb, None, u_perturb, goals, obstacles)
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
        if obstacles.shape[0] > 0 and self.escape==True:
            us.append(u_escape1[0])
            us.append(u_escape2[0])
        # w = self.compute_weight(cost)

        # u = (u_perturb * w.reshape(-1,1,1,1)).sum((0,1))
        us = torch.stack(us, dim=0)
        
        self.dynamic.set_initial_condition({'pos':cur_state.reshape(-1, 1, self.action_dim).repeat(us.shape[0], 1, 1)})
        xs = self.dynamic.integrate_samples(us.reshape(us.shape[0],1,-1,self.action_dim))

        # c_g, w_g = self.goal_cost(xs, goals)
        cost = self.cost_computation(xs, None, us.reshape(us.shape[0],1,-1,self.action_dim), goals, obstacles)
        best_mode = torch.argmin(cost)
        if best_mode > 11:
            print("escape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!")
        u = us[best_mode][0]

        u = self.average_filter(u)
        self.nominal_u[...,:-1,:] = u.reshape(1,1,-1,self.action_dim)[...,1:,:]
        self.nominal_u[...,-1,:] = u.reshape(1,1,-1,self.action_dim)[...,0,:]

        low_cost_traj = []
        for i in range(10):
            # idx = indices[i]
            low_cost_traj.append(xs[i].cpu().numpy())
        low_cost_traj.append(xs[best_mode].cpu().numpy())
        # if cost[best_mode] >= 9990:
        #     u[0] = u[0]*0.0
        # print("planning time:", time.time() - t)
        return u[0], low_cost_traj
        # return v_u

        








        


