import torch.nn as nn
from harl.utils.models_tools import get_active_func
from harl.models.base.entropy_models import EntropyBottleneck
from harl.models.base.Entropy_latent import EntropyBottleneckLatentCodec
from harl.models.base.FactorizedPrior import FactorizedPrior
from harl.models.base.rate_distortion import RateDistortionLoss
from harl.models.base.Round import RoundSTE
import torch
from harl.utils.models_tools import init, get_active_func, get_init_method


class PlainMLP(nn.Module):
    """Plain MLP"""

    def __init__(self, sizes, activation_func, final_activation_func="identity"):
        super().__init__()
        layers = []
        for j in range(len(sizes) - 1):
            act = activation_func if j < len(sizes) - 2 else final_activation_func
            layers += [nn.Linear(sizes[j], sizes[j + 1]), get_active_func(act)]
        self.mlp = nn.Sequential(*layers)
        self.loss = RateDistortionLoss(lmbda=0.01, metric="mse", return_type="all")
        self.round = RoundSTE()

        self.EB = EntropyBottleneck(channels=1)

        self.codec = EntropyBottleneckLatentCodec(entropy_bottleneck=self.EB)
    

    def forward(self, x):
        x = self.mlp(x).unsqueeze(0)
        
        
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

        return fea, bpp_loss, aux_loss
        
