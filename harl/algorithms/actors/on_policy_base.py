"""Base class for on-policy algorithms."""

import torch
from harl.models.policy_models.stochastic_policy import StochasticPolicy
from harl.utils.models_tools import update_linear_schedule
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.models.base.SPS import Predict_Network, Predict_Network_WithID, Predict_ID_obs_tau


class OnPolicyBase:
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        """Initialize Base class.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        # save arguments
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.action_aggregation = args["action_aggregation"]
        self.obs_diversity = args["obs_aware"]
        self.id_diversity = args["id_aware"]
        self.beta = args["beta"]
        self.beta_1 = args["beta_1"]
        self.beta_2 = args["beta_2"]
        obs_shape = get_shape_from_obs_space(obs_space)

        if act_space.__class__.__name__ == "Discrete":
            self.action_dim = act_space.n
        elif act_space.__class__.__name__ == "Box":
            self.action_dim = act_space.shape[0]
        elif act_space.__class__.__name__ == "MultiDiscrete":
            self.multidiscrete_action = True
            self.action_dim = act_space.nvec
        
        self.num_agents = num_agents
        self.predict_withoutID = Predict_Network(obs_shape[0] + self.action_dim, 128, obs_shape[0]).to('cuda')
        self.predict_withID = Predict_Network_WithID(obs_shape[0] + self.action_dim + self.num_agents, 128, obs_shape[0], self.num_agents).to('cuda')


        self.lr = args["lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        # save observation and action spaces
        self.obs_space = obs_space
        self.act_space = act_space
        # create actor network
        self.actor = StochasticPolicy(args, self.obs_space, self.act_space, self.num_agents, self.device)
        # create actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the learning rates.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)

    def get_actions(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions for the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor has RNN layer, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                 (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, action_log_probs, rnn_states_actor, dis_actions = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, action_log_probs, rnn_states_actor, dis_actions

    def evaluate_actions(
        self,
        obs,
        rnn_states_actor,
        action,
        masks,
        available_actions=None,
        active_masks=None,
    ):
        """Get action logprobs, entropy, and distributions for actor update.
        Args:
            obs: (np.ndarray / torch.Tensor) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray / torch.Tensor) if actor has RNN layer, RNN states for actor.
            action: (np.ndarray / torch.Tensor) actions whose log probabilities and entropy to compute.
            masks: (np.ndarray / torch.Tensor) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                    (if None, all actions available)
            active_masks: (np.ndarray / torch.Tensor) denotes whether an agent is active or dead.
        """

        (
            action_log_probs,
            dist_entropy,
            action_distribution,
            bpp_loss,
            kl_loss,
        ) = self.actor.evaluate_actions(
            obs, rnn_states_actor, action, masks, available_actions, active_masks
        )
        return action_log_probs, dist_entropy, action_distribution, bpp_loss, kl_loss

    def act(
        self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False
    ):
        """Compute actions using the given inputs.
        Args:
            obs: (np.ndarray) local agent inputs to the actor.
            rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
            masks: (np.ndarray) denotes points at which RNN states should be reset.
            available_actions: (np.ndarray) denotes which actions are available to agent
                                    (if None, all actions available)
            deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor, _ = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        return actions, rnn_states_actor

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

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        """
        pass

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        """
        pass

    def prep_training(self):
        """Prepare for training."""
        self.actor.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.actor.eval()
