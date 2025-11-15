import torch
import torch.nn as nn
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.models.base.act import ACTLayer
from harl.models.base.SPS import Predict_Network, Predict_Network_WithID, Predict_ID_obs_tau

class StochasticMlpPolicy(nn.Module):
    """Stochastic policy model that only uses MLP network. Outputs actions given observations."""

    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        """Initialize StochasticMlpPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super(StochasticMlpPolicy, self).__init__()
        self.hidden_sizes = args["hidden_sizes"]
        self.args = args
        self.gain = args["gain"]
        self.initialization_method = args["initialization_method"]
        
        if action_space.__class__.__name__ == "Discrete":
            self.action_dim = action_space.n
        elif action_space.__class__.__name__ == "Box":
            self.action_dim = action_space.shape[0]
        elif action_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            self.action_dim = action_space.nvec

        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )
        self.predict_withoutID = Predict_Network(obs_shape[0] + self.action_dim, self.hidden_sizes[0], obs_shape[0])
        self.predict_withID = Predict_Network_WithID(obs_shape[0] + self.action_dim + self.num_agents, self.hidden_sizes[0], obs_shape[0], self.num_agents)
        self.to(device)

    def forward(self, obs, available_actions=None, stochastic=True):
        """Compute actions from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) observation inputs into network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
            stochastic: (bool) whether to sample from action distribution or return the mode.
        Returns:
            actions: (torch.Tensor) actions to take.
        """
        obs = check(obs).to(**self.tpdv)
        deterministic = not stochastic
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features, bpp_loss, aux_loss  = self.base(obs)

        actions, action_log_probs = self.act(
            actor_features, available_actions, deterministic
        )
        return actions, bpp_loss, aux_loss

    def get_logits(self, obs, available_actions=None):
        """Get action logits from the given inputs.
        Args:
            obs: (np.ndarray / torch.Tensor) input to network.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                      (if None, all actions available)
        Returns:
            action_logits: (torch.Tensor) logits of actions for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features, bpp_loss, aux_loss = self.base(obs)

        return self.act.get_logits(actor_features, available_actions), bpp_loss, aux_loss

    def get_log_pi_withoutID(self, inp, out):
        inp = check(inp).to(**self.tpdv)
        out = check(out).to(**self.tpdv)
        pred = self.predict_withoutID.get_log_pi(inp, out)
        return pred

    def update_withoutID(self, inp, out):
        inp = check(inp).to(**self.tpdv)
        out = check(out).to(**self.tpdv)
        loss = self.predict_withoutID.update(inp, out)
        return loss

    def get_log_pi_withID(self, inp, out, add_id):
        inp = check(inp).to(**self.tpdv)
        out = check(out).to(**self.tpdv)
        pred = self.predict_withID.get_log_pi(inp, out, add_id)
        return pred

    def update_withID(self, inp, out, add_id):
        inp = check(inp).to(**self.tpdv)
        out = check(out).to(**self.tpdv)
        loss = self.predict_withID.update(inp, out, add_id)
        return loss
