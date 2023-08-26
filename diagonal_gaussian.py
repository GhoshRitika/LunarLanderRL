import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class DiagGaussian(nn.Module):
    """Diagonal Gaussian distribution.

    Mean is parameterized as a linear function of the features.
    logstd is torch.Parameter by default, but can also be a linear function
    of the features.
    """

    def __init__(self, nin, nout, constant_log_std=True, log_std_min=-20,
                 log_std_max=2):
        """Init.

        Args:
            nin  (int): dimensionality of the input
            nout (int): number of categories
            constant_log_std (bool): If False, logstd will be a linear function
                                     of the features.

        """
        super().__init__()
        self.constant_log_std = constant_log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc_mean = nn.Linear(nin, nout).float()
        nn.init.orthogonal_(self.fc_mean.weight.data, gain=0.0)
        nn.init.constant_(self.fc_mean.bias.data, 0)
        if constant_log_std:
            self.logstd = nn.Parameter(torch.zeros(nout)).float()
        else:
            self.fc_logstd = nn.Linear(nin, nout)
            nn.init.orthogonal_(self.fc_logstd.weight.data, gain=0.0)
            nn.init.constant_(self.fc_logstd.bias.data, 0)

    def forward(self, x, return_logstd=False):
        """Forward.

        Args:
            x (torch.Tensor): vectors of length nin
        Returns:
            dist (torch.distributions.Normal): normal distribution

        """
        mean = self.fc_mean(x)
        if self.constant_log_std:
            logstd = torch.clamp(self.logstd, self.log_std_min,
                                 self.log_std_max)
        else:
            logstd = torch.clamp(self.fc_logstd(x), self.log_std_min,
                                 self.log_std_max)
        if return_logstd:
            return Normal(mean, logstd.exp()), logstd
        else:
            return Normal(mean, logstd.exp()), mean, logstd.exp()