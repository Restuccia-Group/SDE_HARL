"""HAA2C algorithm."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from harl.utils.envs_tools import check
from harl.utils.models_tools import get_grad_norm
from harl.algorithms.actors.on_policy_base import OnPolicyBase


class HAA2C(OnPolicyBase):
    def __init__(self, args, obs_space, act_space, num_agents, device=torch.device("cpu")):
        """Initialize HAA2C algorithm.
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(HAA2C, self).__init__(args, obs_space, act_space, num_agents, device)

        self.a2c_epoch = args["a2c_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

    def update(self, sample):
        """Update actor network.
        Args:
            sample: (Tuple) contains data batch with which to update networks.
        Returns:
            policy_loss: (torch.Tensor) actor(policy) loss value.
            dist_entropy: (torch.Tensor) action entropies.
            actor_grad_norm: (torch.Tensor) gradient norm from actor update.
            imp_weights: (torch.Tensor) importance sampling weights.
        """
        (
            obs_current_batch,
            obs_next_batch, #notice: this is input of actor
            obs_next_share_batch,
            rnn_states_batch,
            actions_batch,
            actions_onehot_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            dis_actions_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)

        # Reshape to do evaluations for all steps in a single forward pass
        action_log_probs, dist_entropy, action_distribution, bpp_loss, aux_loss = self.evaluate_actions(
            obs_next_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )
        #training SPS

        if self.obs_diversity:
            add_id = torch.eye(obs_current_batch.shape[1]).to(self.device).unsqueeze(0).expand(
                        [obs_current_batch.shape[0], obs_current_batch.shape[1], obs_current_batch.shape[1]])

            if isinstance(obs_current_batch, np.ndarray):
                obs_current_batch = torch.from_numpy(obs_current_batch).float().to(self.device).requires_grad_(True)
            else:
                obs_current_batch = obs_current_batch.requires_grad_(True)

            if isinstance(actions_onehot_batch, np.ndarray):
                actions_onehot_batch = torch.from_numpy(actions_onehot_batch).float().to(self.device).requires_grad_(True)
            else:
                actions_onehot_batch = actions_onehot_batch.requires_grad_(True)

            if isinstance(obs_next_share_batch, np.ndarray):
                obs_next_share_batch = torch.from_numpy(obs_next_share_batch).float().to(self.device).requires_grad_(True)
            else:
                obs_next_share_batch = obs_next_share_batch.requires_grad_(True)

            intrinsic_input = torch.cat([
                                obs_current_batch,
                                actions_onehot_batch
                            ], dim=-1)

            loss_withoutID = self.update_withoutID(
                                intrinsic_input, obs_next_share_batch
                                )

            loss_withID = self.update_withID(
                                intrinsic_input, obs_next_share_batch, add_id
                                )
            
            with torch.no_grad():

                log_p_o = self.get_log_pi_withoutID(
                                intrinsic_input, obs_next_share_batch)
        
                            
                log_q_o = self.get_log_pi_withID(
                                intrinsic_input, obs_next_share_batch, add_id)

                obs_diverge = self.beta_1 * log_q_o - log_p_o

        if self.id_diversity:

            with torch.no_grad():
                
                if isinstance(dis_actions_batch, np.ndarray):
                    dis_actions_batch = torch.from_numpy(dis_actions_batch).float().to(self.device).requires_grad_(True)
                else:
                    dis_actions_batch = dis_actions_batch.requires_grad_(True)

                mean_p = torch.stack(
                        [F.softmax(l, dim=-1) for l in dis_actions_batch],
                        dim=0
                        ).mean(dim=1)
            
            #q_pi = F.softmax(self.beta_1 * logits[agent_id], dim=-1).clamp(min=1e-8)
            q_pi = F.softmax(action_distribution.logits, dim=-1).clamp(min=1e-8)
            
            pi_diverge = (
                q_pi * (
                    torch.log(q_pi) - torch.log(mean_p)
                )
            ).sum(dim=-1, keepdim=True)
        
        # actor update
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        surr = imp_weights * adv_targ

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * surr, dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * surr, dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        if self.id_diversity and self.obs_diversity:
            intrinsic_rewards = self.beta_2 * pi_diverge #+ obs_diverge
        elif self.id_diversity and not self.obs_diversity:
            intrinsic_rewards = self.beta_2 * pi_diverge
        elif not self.id_diversity and self.obs_diversity:
            intrinsic_rewards = obs_diverge
        else:
            intrinsic_rewards = None

        if intrinsic_rewards is not None:
            intrinsic_rewards = (intrinsic_rewards * active_masks_batch).sum() / active_masks_batch.sum()
            policy_loss += self.beta * intrinsic_rewards


        self.actor_optimizer.zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()  # add entropy term

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, actor_buffer, advantages, state_type):
        """Perform a training update using minibatch GD.
        Args:
            actor_buffer: (OnPolicyActorBuffer) buffer containing training data related to actor.
            advantages: (np.ndarray) advantages.
            state_type: (str) type of state.
        Returns:
            train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0

        if np.all(actor_buffer.active_masks[:-1] == 0.0):
            return train_info

        if state_type == "EP":
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.a2c_epoch):
            if self.use_recurrent_policy:
                data_generator = actor_buffer.recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer.naive_recurrent_generator_actor(
                    advantages, self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer.feed_forward_generator_actor(
                    advantages, self.actor_num_mini_batch
                )

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()

        num_updates = self.a2c_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
