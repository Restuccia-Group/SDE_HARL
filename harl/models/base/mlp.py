import torch.nn as nn
import torch
from harl.utils.models_tools import init, get_active_func, get_init_method
from harl.models.base.entropy_models import EntropyBottleneck
from harl.models.base.Entropy_latent import EntropyBottleneckLatentCodec
from harl.models.base.FactorizedPrior import FactorizedPrior
from harl.models.base.rate_distortion import RateDistortionLoss
from harl.models.base.Round import RoundSTE

"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    """A MLP base module."""

    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()

        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]

        obs_dim = obs_shape[0]

        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func
        )
        self.loss = RateDistortionLoss(lmbda=0.01, metric="mse", return_type="all")
        self.round = RoundSTE()

        self.EB = EntropyBottleneck(channels=1)

        self.codec = EntropyBottleneckLatentCodec(entropy_bottleneck=self.EB)
    

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        
        x = self.mlp(x).unsqueeze(0)
        #x = self.mlp(x)
        
        
        if self.training:
            out = self.codec(x) 
            bpp_loss = self.loss(out, x)["bpp_loss"] 
                
            aux_loss = self.EB.loss()
            fea = self.round(x).squeeze(0)
    
        else:
            self.EB.update(force=True)
            bit = self.codec.compress(x.permute(1, 0, 2)) 
            y_hats = bit["y_hat"] 
            fea = x.view(y_hats.size(0), -1)
            bpp_loss, aux_loss = None, None
        x = x.squeeze(0)

        return x, bpp_loss, aux_loss
        