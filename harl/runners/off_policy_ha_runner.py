"""Runner for off-policy HARL algorithms."""
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from harl.runners.off_policy_base_runner import OffPolicyBaseRunner


class OffPolicyHARunner(OffPolicyBaseRunner):
    """Runner for off-policy HA algorithms."""

    def train(self):
        """Train the model"""
        self.total_it += 1
        data = self.buffer.sample()
        (
            sp_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_obs,  # (n_agents, batch_size, dim)
            sp_actions,  # (n_agents, batch_size, dim)
            sp_available_actions,  # (n_agents, batch_size, dim)
            sp_reward,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_done,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_valid_transition,  # (n_agents, batch_size, 1)
            sp_term,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
            sp_next_share_obs,  # EP: (batch_size, dim), FP: (n_agents * batch_size, dim)
            sp_next_obs,  # (n_agents, batch_size, dim)
            sp_next_available_actions,  # (n_agents, batch_size, dim)
            sp_gamma,  # EP: (batch_size, 1), FP: (n_agents * batch_size, 1)
        ) = data
        # train critic
        self.critic.turn_on_grad()
        if self.args["algo"] == "hasac":
            next_actions = []
            next_logp_actions = []
            for agent_id in range(self.num_agents):
                next_action, next_logp_action, _, _, _ = self.actor[
                    agent_id
                ].get_actions_with_logprobs(
                    sp_next_obs[agent_id],
                    sp_next_available_actions[agent_id]
                    if sp_next_available_actions is not None
                    else None,
                )
                next_actions.append(next_action)
                next_logp_actions.append(next_logp_action)
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_valid_transition,
                sp_term,
                sp_next_share_obs,
                next_actions,
                next_logp_actions,
                sp_gamma,
                self.value_normalizer,
            )
        else:
            next_actions = []
            for agent_id in range(self.num_agents):
                next_actions.append(
                    self.actor[agent_id].get_target_actions(sp_next_obs[agent_id])
                )
            self.critic.train(
                sp_share_obs,
                sp_actions,
                sp_reward,
                sp_done,
                sp_term,
                sp_next_share_obs,
                next_actions,
                sp_gamma,
            )
        self.critic.turn_off_grad()
        sp_valid_transition = torch.tensor(sp_valid_transition, device=self.device)
        if self.total_it % self.policy_freq == 0:
            # train actors
            if self.args["algo"] == "hasac":
                actions = []
                logp_actions = []
                logits = []
                with torch.no_grad():
                    for agent_id in range(self.num_agents):
                        action, logp_action, logit, _, _ = self.actor[
                            agent_id
                        ].get_actions_with_logprobs(
                            sp_obs[agent_id],
                            sp_available_actions[agent_id]
                            if sp_available_actions is not None
                            else None,
                        )
                        actions.append(action)
                        logp_actions.append(logp_action)
                        logits.append(logit)
                    
                    # mean_p = torch.stack(
                    #     [F.softmax(l, dim=-1) for l in logits],
                    #     dim=0
                    # ).mean(dim=0)

                
                # actions shape: (n_agents, batch_size, dim)
                # logp_actions shape: (n_agents, batch_size, 1)
                if self.fixed_order:
                    agent_order = list(range(self.num_agents))
                else:
                    #agent_order = list(np.random.permutation(self.num_agents)) #randomly
                    # ---- STEP 1: Learnable scores for agents ----
                    if not hasattr(self, "agent_score_logits"):
                        self.agent_score_logits = nn.Parameter(torch.randn(self.num_agents))
                        self.agent_score_logits.requires_grad = True
                        self.order_optimizer = torch.optim.Adam([self.agent_score_logits], lr=1e-2)

                    agent_scores = self.agent_score_logits  # shape: [num_agents], learnable

                    # ---- STEP 2: Differentiable order approximation using Gumbel-Softmax ----
                    def gumbel_sort(logits, tau=1.0):
                        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
                        noisy_logits = logits + gumbel_noise
                        probs = torch.softmax(noisy_logits / tau, dim=0)
                        return probs

                    soft_permutation = gumbel_sort(agent_scores, tau=0.5)  # shape: [num_agents]

                    # ---- STEP 3: Get hard execution order (non-diff for now) ----
                    agent_order = torch.argsort(-soft_permutation).tolist()  # [0, 2, 1], etc.
                    #print(agent_order)

                    # ---- STEP 4: Optional logging for later reward signal ----
                    self.last_soft_ranks = soft_permutation
                    self.last_agent_scores = agent_scores

                # ---- Run policy update loop ----
                total_actor_loss = []                    
                for agent_id in agent_order:
                    self.actor[agent_id].turn_on_grad()
                    # train this agent
                    actions[agent_id], logp_actions[agent_id], logits[agent_id], bpp_loss, aux_loss = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                    
                    #training SPS

                    if self.obs_diversity:
                        add_id = torch.eye(self.num_agents).to(self.device).unsqueeze(1).expand(
                                    [sp_obs.shape[0], sp_obs.shape[1], self.num_agents])

                        actions_onehot = torch.stack(actions, dim=0)


                        if isinstance(sp_obs, np.ndarray):
                            sp_obs_tensor = torch.from_numpy(sp_obs).float().to(self.device).requires_grad_(True)
                        else:
                            sp_obs_tensor = sp_obs.requires_grad_(True)

                        intrinsic_input = torch.cat([
                                            sp_obs_tensor,
                                            actions_onehot
                                        ], dim=-1)

                        loss_withoutID = self.actor[agent_id].update_withoutID(
                                            intrinsic_input, sp_next_obs
                                            )

                        loss_withID = self.actor[agent_id].update_withID(
                                        intrinsic_input, sp_next_obs, add_id)
                        
                        with torch.no_grad():
                            log_p_o = self.actor[agent_id].get_log_pi_withoutID(
                                            intrinsic_input, sp_next_obs)
                    
                                        
                            log_q_o = self.actor[agent_id].get_log_pi_withID(
                                            intrinsic_input, sp_next_obs, add_id)

                            obs_diverge = self.beta_1 * log_q_o - log_p_o

                    if self.id_diversity:

                        with torch.no_grad():
                            mean_p = torch.stack(
                                [F.softmax(l, dim=-1) for l in logits],
                                dim=0
                            ).mean(dim=0)
                        
                        #q_pi = F.softmax(self.beta_1 * logits[agent_id], dim=-1).clamp(min=1e-8)
                        q_pi = F.softmax(logits[agent_id], dim=-1).clamp(min=1e-8)
                        
                        pi_diverge = (
                            q_pi * (
                                torch.log(q_pi) - torch.log(mean_p)
                            )
                        ).sum(dim=-1, keepdim=True)
                        
                        
                    if self.state_type == "EP":
                        logp_action = logp_actions[agent_id]
                        actions_t = torch.cat(actions, dim=-1)
                        mask = sp_valid_transition[agent_id]
                    elif self.state_type == "FP":
                        logp_action = torch.tile(
                            logp_actions[agent_id], (self.num_agents, 1)
                        )
                        actions_t = torch.tile(
                            torch.cat(actions, dim=-1), (self.num_agents, 1)
                        )
                        mask = torch.tile(sp_valid_transition[agent_id], (self.num_agents, 1))

                    value_pred = self.critic.get_values(sp_share_obs, actions_t)

                    if self.algo_args["algo"]["use_policy_active_masks"]:
                        actor_loss = (
                            -torch.sum((value_pred - self.alpha[agent_id] * logp_action) * mask)
                            / mask.sum()
                        ) 

                    else:
                        actor_loss = -torch.mean(value_pred - self.alpha[agent_id] * logp_action)

                    if self.id_diversity and self.obs_diversity:
                        intrinsic_rewards = self.beta_2 * pi_diverge + obs_diverge
                    elif self.id_diversity and not self.obs_diversity:
                        intrinsic_rewards = self.beta_2 * pi_diverge
                    elif not self.id_diversity and self.obs_diversity:
                        intrinsic_rewards = obs_diverge
                    else:
                        intrinsic_rewards = None

                    if intrinsic_rewards is not None:
                        intrinsic_rewards = (intrinsic_rewards * mask).sum() / mask.sum()
                        actor_loss += self.beta * intrinsic_rewards
                    

                    self.actor[agent_id].actor_optimizer.zero_grad()
                    actor_loss.backward()
                   
                    self.actor[agent_id].actor_optimizer.step()
                    self.actor[agent_id].turn_off_grad()
                    # train this agent's alpha
                    if self.algo_args["algo"]["auto_alpha"]:
                        log_prob = (
                            logp_actions[agent_id].detach()
                            + self.target_entropy[agent_id]
                        )
                        alpha_loss = -(self.log_alpha[agent_id] * log_prob).mean()
                        self.alpha_optimizer[agent_id].zero_grad()
                        alpha_loss.backward()
                        self.alpha_optimizer[agent_id].step()
                        self.alpha[agent_id] = torch.exp(
                            self.log_alpha[agent_id].detach()
                        )
                    actions[agent_id], _, logits[agent_id], _, _ = self.actor[
                        agent_id
                    ].get_actions_with_logprobs(
                        sp_obs[agent_id],
                        sp_available_actions[agent_id]
                        if sp_available_actions is not None
                        else None,
                    )
                    actor_loss_order = actor_loss.clone()
                    total_actor_loss.append(actor_loss_order)
                # train critic's alpha
                if self.algo_args["algo"]["auto_alpha"]:
                    self.critic.update_alpha(logp_actions, np.sum(self.target_entropy))
                # ---- After loop: backprop through ordering based on surrogate reward ----
                if not self.fixed_order:
                    # Detach total_actor_loss to avoid second backward error
                    total_loss_tensor = torch.stack([l.detach() for l in total_actor_loss])
                    soft_permutation = soft_permutation.to(total_loss_tensor.device)
                    surrogate_reward = -torch.sum(total_loss_tensor * soft_permutation)  # Use soft order weights

                    self.order_optimizer.zero_grad()
                    surrogate_reward.backward()
                    self.order_optimizer.step()
                if self.total_it % 200 == 0:  # add train_step tracking if needed
                    print("agent_score_logits:", agent_order)

            else:
                if self.args["algo"] == "had3qn":
                    actions = []
                    logits = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            action, logit, _, _ = self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            actions.append(action)
                            logits.append(logit)
                   
                    # actions shape: (n_agents, batch_size, 1)
                    update_actions, get_values = self.critic.train_values(
                        sp_share_obs, actions
                    )
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # actor preds
                        actor_values, logits[agent_id], bpp_loss, aux_loss = self.actor[agent_id].train_values(
                            sp_obs[agent_id], actions[agent_id]
                        )
                        # critic preds
                        critic_values = get_values()
                        #training SPS

                        if self.obs_diversity:
                            add_id = torch.eye(self.num_agents).to(self.device).unsqueeze(1).expand(
                                        [sp_obs.shape[0], sp_obs.shape[1], self.num_agents])

                            actions_onehot = torch.stack(actions, dim=0)


                            if isinstance(sp_obs, np.ndarray):
                                sp_obs_tensor = torch.from_numpy(sp_obs).float().to(self.device).requires_grad_(True)
                            else:
                                sp_obs_tensor = sp_obs.requires_grad_(True)

                            intrinsic_input = torch.cat([
                                                sp_obs_tensor,
                                                actions_onehot
                                            ], dim=-1)

                            loss_withoutID = self.actor[agent_id].update_withoutID(
                                                intrinsic_input, sp_next_obs
                                                )

                            loss_withID = self.actor[agent_id].update_withID(
                                            intrinsic_input, sp_next_obs, add_id)
                            
                            with torch.no_grad():
                                log_p_o = self.actor[agent_id].get_log_pi_withoutID(
                                                intrinsic_input, sp_next_obs)
                        
                                            
                                log_q_o = self.actor[agent_id].get_log_pi_withID(
                                                intrinsic_input, sp_next_obs, add_id)

                                obs_diverge = self.beta_1 * log_q_o - log_p_o

                        if self.id_diversity:

                            with torch.no_grad():
                                mean_p = torch.stack(
                                    [F.softmax(l, dim=-1) for l in logits],
                                    dim=0
                                ).mean(dim=0)
                            
                                #q_pi = F.softmax(self.beta_1 * logits[agent_id], dim=-1).clamp(min=1e-8)
                                q_pi = F.softmax(logits[agent_id], dim=-1).clamp(min=1e-8)
                                
                                pi_diverge = (
                                    q_pi * (
                                        torch.log(q_pi) - torch.log(mean_p)
                                    )
                                ).sum(dim=-1, keepdim=True)
                        
                        if self.state_type == "EP":
                            mask = sp_valid_transition[agent_id]
                        elif self.state_type == "FP":
                            mask = torch.tile(sp_valid_transition[agent_id], (self.num_agents, 1))

                        if self.id_diversity and self.obs_diversity:
                            intrinsic_rewards = self.beta_2 * pi_diverge #+ obs_diverge
                        elif self.id_diversity and not self.obs_diversity:
                            intrinsic_rewards = self.beta_2 * pi_diverge
                        elif not self.id_diversity and self.obs_diversity:
                            intrinsic_rewards = obs_diverge
                        else:
                            intrinsic_rewards = None

                        if intrinsic_rewards is not None:
                            intrinsic_rewards = (intrinsic_rewards * mask).sum() / mask.sum()
                            actor_loss += self.beta * intrinsic_rewards

                        # update
                        actor_loss = torch.mean(F.mse_loss(actor_values, critic_values))
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        #print(actor_loss)
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        update_actions(agent_id)
                else:
                    actions = []
                    with torch.no_grad():
                        for agent_id in range(self.num_agents):
                            actions.append(
                                self.actor[agent_id].get_actions(
                                    sp_obs[agent_id], False
                                )
                            )
                    # actions shape: (n_agents, batch_size, dim)
                    if self.fixed_order:
                        agent_order = list(range(self.num_agents))
                    else:
                        agent_order = list(np.random.permutation(self.num_agents))
                    for agent_id in agent_order:
                        self.actor[agent_id].turn_on_grad()
                        # train this agent
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                        actions_t = torch.cat(actions, dim=-1)
                        value_pred = self.critic.get_values(sp_share_obs, actions_t)
                        actor_loss = -torch.mean(value_pred)
                        self.actor[agent_id].actor_optimizer.zero_grad()
                        actor_loss.backward()
                        self.actor[agent_id].actor_optimizer.step()
                        self.actor[agent_id].turn_off_grad()
                        actions[agent_id] = self.actor[agent_id].get_actions(
                            sp_obs[agent_id], False
                        )
                # soft update
                for agent_id in range(self.num_agents):
                    self.actor[agent_id].soft_update()
            self.critic.soft_update()
