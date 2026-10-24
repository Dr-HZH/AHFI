import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .gnn_backbone import GNNBackbone

class ScaleInvariantMonitor(nn.Module):
    """
    Scale-Invariant Relative Constraint Monitor
    
    Instead of using an absolute budget (e.g., 0.1), this monitor enforces a relative constraint:
    current_risk <= max(gamma * historical_moving_average, min_risk)
    
    This solves the scale issue across different datasets and layers.
    """
    def __init__(self, momentum=0.9, gamma=0.1, min_risk=0.002, d=None):
        super().__init__()
        self.momentum = momentum
        self.gamma = gamma
        self.min_risk = min_risk

        self.register_buffer('risk_moving_avg', torch.tensor(1.0))


        self.last_risk = 0.0
        self.last_target = 0.0
        self.is_violated = False

    def forward(self, h, s):

        if h.dim() > 2: h = h.mean(dim=1)
        h = h / (h.norm(p=2, dim=1, keepdim=True) + 1e-9)


        s0_mask, s1_mask = (s == 0), (s == 1)
        if not (s0_mask.any() and s1_mask.any()):
            return torch.tensor(0.0, device=h.device), torch.tensor(0.0, device=h.device)

        mu0 = h[s0_mask].sum(0)
        mu1 = h[s1_mask].sum(0)
        mu0 = mu0 / (mu0.norm(p=2) + 1e-9)
        mu1 = mu1 / (mu1.norm(p=2) + 1e-9)

        # Inner product of spherical centroids, Eqs. (5)-(6).
        cos_sim = torch.dot(mu0, mu1)

        cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)




        current_risk_angle = torch.acos(cos_sim) / torch.pi


        if self.training:
            with torch.no_grad():
                self.risk_moving_avg.mul_(self.momentum).add_(
                    current_risk_angle.detach() * (1 - self.momentum)
                )




        target = max(self.gamma * self.risk_moving_avg, self.min_risk)
        if isinstance(target, torch.Tensor):
             target_val = target
        else:
             target_val = torch.tensor(target, device=current_risk_angle.device)


        violation = F.relu(torch.log(current_risk_angle + 1e-9) - torch.log(target_val + 1e-9))


        self.last_risk = current_risk_angle.item()
        self.last_target = target_val.item()
        self.is_violated = (current_risk_angle > target_val).item()

        return current_risk_angle, violation

class LearnableProjector(nn.Module):
    """
    Learnable affine projection
    
    A learnable linear transformation that rotates/scales the feature space
    to satisfy fairness constraints, guided by the dual variable lambda.
    """
    def __init__(self, dim, lambda_decay=0.01):
        super().__init__()
        self.lambda_decay = lambda_decay

        self.linear = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(dim))
            self.linear.bias.zero_()
            


        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.linear(x)

    def get_proxy_loss(self, violation):


        lambda_val = torch.exp(self.log_lambda)
        return lambda_val.detach() * violation

    def get_dual_loss(self, violation):


        lambda_val = torch.exp(self.log_lambda)
        return -lambda_val * violation.detach() + self.lambda_decay * lambda_val.square()

class Adaptive_Hierarchical(nn.Module):
    """
    Adaptive Hierarchical Fairness Intervention Framework
    
    Features:
    1. Scale-Invariant Relative Constraints (Gamma)
    2. Learnable affine projections
    3. Primal-Dual Optimization
    """
    def __init__(self, nfeat, nhid, nclass, num_nodes, arch='GCN', heads=2, dropout=0.5, gamma=0.1, lambda_decay=0.01, ema_momentum=0.9):
        """
        Args:
            num_nodes (int): Total number of nodes in the dataset, number of monitored nodes for the finite-sample floor
        """
        super().__init__()


        self.gnn = GNNBackbone(arch=arch, nfeat=nfeat, nhid=nhid, nclass=nclass,
                              heads=heads, dropout=dropout)

        # Finite-sample stabilization scale from Eq. (9), not a confidence bound.
        adaptive_risk = 1.0 / math.sqrt(num_nodes)


        self.monitors = nn.ModuleDict({
            'input': ScaleInvariantMonitor(momentum=ema_momentum, gamma=gamma, min_risk=adaptive_risk, d=nfeat),
            'rep':   ScaleInvariantMonitor(momentum=ema_momentum, gamma=gamma, min_risk=adaptive_risk, d=nhid),
            'out':   ScaleInvariantMonitor(momentum=ema_momentum, gamma=gamma, min_risk=adaptive_risk, d=nclass)
        })
        

        self.projectors = nn.ModuleDict({
            'input': LearnableProjector(nfeat, lambda_decay),
            'rep':   LearnableProjector(nhid, lambda_decay),
            'out':   LearnableProjector(nclass, lambda_decay)
        })

    def forward(self, g, x, s, y=None):

        x_proj = self.projectors['input'](x)
        input_risk, input_violation = self.monitors['input'](x_proj, s)



        h = self.gnn.get_first_layer_output(g, x_proj)
        h_proj = self.projectors['rep'](h)
        rep_risk, rep_violation = self.monitors['rep'](h_proj, s)



        logits = self.gnn.forward_from_intermediate(g, h_proj)
        logits_proj = self.projectors['out'](logits)
        out_risk, out_violation = self.monitors['out'](logits_proj, s)



        primal_fair_loss = (
            self.projectors['input'].get_proxy_loss(input_violation) +
            self.projectors['rep'].get_proxy_loss(rep_violation) +
            self.projectors['out'].get_proxy_loss(out_violation)
        )


        dual_loss = (
            self.projectors['input'].get_dual_loss(input_violation) +
            self.projectors['rep'].get_dual_loss(rep_violation) +
            self.projectors['out'].get_dual_loss(out_violation)
        )

        if self.training:

            return logits_proj, primal_fair_loss, dual_loss
        else:

            info = {
                'input_risk': input_risk.item(),
                'rep_risk': rep_risk.item(),
                'out_risk': out_risk.item(),
                'lambda_in': torch.exp(self.projectors['input'].log_lambda).item(),
                'lambda_rep': torch.exp(self.projectors['rep'].log_lambda).item(),
                'lambda_out': torch.exp(self.projectors['out'].log_lambda).item()
            }
            return logits_proj, info
