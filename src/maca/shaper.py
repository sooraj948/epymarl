import numpy as np
from typing import List, Tuple, Optional
from .reducers import BaseDimensionalityReducer
from .density import BaseDensityEstimator

class CounterfactualRewardShaper:
    def __init__(self, baseline_mode: str = 'medoid', 
                 per_agent_dim: int = 25,
                 n_agents: int = 2,
                 clip_val: float = 1.0,
                 n_samples: int = 16):
        """
        Args:
            baseline_mode: How to construct v_bar_i
                - 'zero_padding': replace agent i's portion with zeros (represents inaction)
                - 'mean_loss': replace with running mean of losing agents' behaviors
                - 'mean_all': replace with running mean of all agents' behaviors
                - 'medoid': replace with agent i's portion of the medoid losing signature
                - 'random_valid': replace with agent i's portion of a randomly sampled losing signature
                - 'multi_sample': replace with average counterfactual score across K randomly sampled losing signatures
            per_agent_dim: dimensionality of each agent's behavioral vector
            n_agents: number of agents
            clip_val: clipping range for intrinsic rewards
            n_samples: number of loss samples to average over when baseline_mode == 'multi_sample'
        """
        self.baseline_mode = baseline_mode
        self.per_agent_dim = per_agent_dim
        self.n_agents = n_agents
        self.clip_val = clip_val
        self.n_samples = n_samples
        self._loss_mean = None
        self._all_mean = None
        self._medoid_loss = None
        self._medoid_idx = None
        self._X_win = None
        self._X_loss = None
        self._loss_trajs = None
        self.running_max_abs = None  # Properly initialize to None to avoid transient overshoot
        # self.running_max_abs = 1.0
    def update_baselines(self, X_win: np.ndarray, X_loss: np.ndarray, loss_trajs: list = None):
        """Update the baseline vectors and raw trajectories from current buffer data."""
        self._X_win = X_win
        self._X_loss = X_loss
        self._loss_trajs = loss_trajs
        
        if len(X_loss) > 0:
            # X_loss shape: (n_episodes, n_agents * per_agent_dim)
            n_ep, dim = X_loss.shape
            reshaped_loss = X_loss.reshape(-1, self.per_agent_dim)
            self._loss_mean = np.mean(reshaped_loss, axis=0)
            
            # Compute medoid joint signature in loss buffer
            if self.baseline_mode == 'medoid':
                if len(X_loss) == 1:
                    self._medoid_loss = X_loss[0]
                    self._medoid_idx = 0
                else:
                    # Pairwise squared Euclidean distances between joint signatures (algebraic expansion for efficiency)
                    X_loss_sq = np.sum(X_loss**2, axis=-1, keepdims=True)
                    dists = X_loss_sq + X_loss_sq.T - 2 * np.dot(X_loss, X_loss.T)
                    dists = np.clip(dists, 0, None)  # numerical stability
                    self._medoid_idx = np.argmin(np.sum(dists, axis=1))
                    self._medoid_loss = X_loss[self._medoid_idx]
            
        all_X = []
        if len(X_win) > 0:
            all_X.append(X_win)
        if len(X_loss) > 0:
            all_X.append(X_loss)
            
        if all_X:
            all_X_concat = np.concatenate(all_X, axis=0)
            reshaped_all = all_X_concat.reshape(-1, self.per_agent_dim)
            self._all_mean = np.mean(reshaped_all, axis=0)
    
    def _construct_counterfactual(self, joint_signature: np.ndarray, 
                                   agent_idx: int) -> np.ndarray:
        """Replace agent_idx's portion of the joint signature with baseline (signature-level)."""
        start = agent_idx * self.per_agent_dim
        end = start + self.per_agent_dim
        
        counterfactual = joint_signature.copy()
        if self.baseline_mode == 'zero_padding':
            counterfactual[start:end] = 0.0
        elif self.baseline_mode == 'mean_loss':
            if self._loss_mean is not None:
                counterfactual[start:end] = self._loss_mean
            else:
                counterfactual[start:end] = 0.0
        elif self.baseline_mode == 'mean_all':
            if self._all_mean is not None:
                counterfactual[start:end] = self._all_mean
            else:
                counterfactual[start:end] = 0.0
        elif self.baseline_mode == 'medoid':
            if self._medoid_loss is not None:
                counterfactual[start:end] = self._medoid_loss[start:end]
            else:
                counterfactual[start:end] = 0.0
        elif self.baseline_mode in ('random_valid', 'multi_sample'):
            if self._X_loss is not None and len(self._X_loss) > 0:
                rand_idx = np.random.choice(len(self._X_loss))
                rand_sig = self._X_loss[rand_idx]
                counterfactual[start:end] = rand_sig[start:end]
            else:
                counterfactual[start:end] = 0.0
        
        return counterfactual

    def _construct_counterfactual_signature_for_loss_idx(self,
                                                          episode_trajectories: List[List],
                                                          agent_idx: int,
                                                          loss_idx: int,
                                                          encoder,
                                                          config,
                                                          agent_signatures: List[np.ndarray] = None) -> np.ndarray:
        """Construct counterfactual joint signature replacing agent_idx with loss_idx's trajectory/signature."""
        T_win = len(episode_trajectories[0])
        baseline_agent_traj = None
        if self._loss_trajs is not None and 0 <= loss_idx < len(self._loss_trajs):
            baseline_agent_traj = self._loss_trajs[loss_idx][agent_idx]
        
        if baseline_agent_traj is not None:
            truncated_traj = baseline_agent_traj[:T_win]
            cf_agent_sig = encoder._create_behavioral_signature(truncated_traj, agent_idx, config)
            cf_joint_sig = []
            for i in range(self.n_agents):
                if i == agent_idx:
                    cf_joint_sig.append(cf_agent_sig)
                elif agent_signatures is not None:
                    cf_joint_sig.append(agent_signatures[i])
                else:
                    true_agent_sig = encoder._create_behavioral_signature(episode_trajectories[i], i, config)
                    cf_joint_sig.append(true_agent_sig)
            return np.concatenate(cf_joint_sig)
        
        if agent_signatures is not None:
            true_joint_sig = np.concatenate(agent_signatures)
        else:
            true_joint_sig = encoder.encode(episode_trajectories, self.n_agents, config)
            
        counterfactual = true_joint_sig.copy()
        start = agent_idx * self.per_agent_dim
        end = start + self.per_agent_dim
        if self._X_loss is not None and 0 <= loss_idx < len(self._X_loss):
            counterfactual[start:end] = self._X_loss[loss_idx][start:end]
        else:
            counterfactual[start:end] = 0.0
        return counterfactual

    def _construct_counterfactual_signature(self, 
                                            episode_trajectories: List[List],
                                            agent_idx: int,
                                            encoder,
                                            config,
                                            agent_signatures: List[np.ndarray] = None) -> np.ndarray:
        """
        Constructs the counterfactual joint signature for agent_idx by truncating
        a loss trajectory to the length of the current episode and encoding it.
        
        Args:
            agent_signatures: Optional pre-computed per-agent signatures to avoid
                              redundant re-encoding of unchanged agents.
        """
        T_win = len(episode_trajectories[0])
        baseline_agent_traj = None
        
        if self.baseline_mode == 'medoid' and self._medoid_idx is not None and self._loss_trajs is not None:
            # Get agent_idx's trajectory from the medoid loss episode
            medoid_loss_traj = self._loss_trajs[self._medoid_idx]
            baseline_agent_traj = medoid_loss_traj[agent_idx]
        elif self.baseline_mode in ('random_valid', 'multi_sample') and self._loss_trajs is not None and len(self._loss_trajs) > 0:
            # Get agent_idx's trajectory from a random loss episode
            rand_idx = np.random.choice(len(self._loss_trajs))
            rand_loss_traj = self._loss_trajs[rand_idx]
            baseline_agent_traj = rand_loss_traj[agent_idx]
            
        if baseline_agent_traj is not None:
            # Truncate raw trajectory of the baseline agent to match win length
            truncated_traj = baseline_agent_traj[:T_win]
            
            # Encode agent_idx's truncated baseline signature
            cf_agent_sig = encoder._create_behavioral_signature(truncated_traj, agent_idx, config)
            
            # Construct the counterfactual joint signature using cached per-agent sigs
            cf_joint_sig = []
            for i in range(self.n_agents):
                if i == agent_idx:
                    cf_joint_sig.append(cf_agent_sig)
                elif agent_signatures is not None:
                    # Reuse cached signature — avoids redundant re-encoding
                    cf_joint_sig.append(agent_signatures[i])
                else:
                    # Fallback: re-encode (original behavior)
                    true_agent_sig = encoder._create_behavioral_signature(episode_trajectories[i], i, config)
                    cf_joint_sig.append(true_agent_sig)
            return np.concatenate(cf_joint_sig)
            
        # Fallback to signature-level baseline if raw trajectories are unavailable
        if agent_signatures is not None:
            true_joint_sig = np.concatenate(agent_signatures)
        else:
            true_joint_sig = encoder.encode(episode_trajectories, self.n_agents, config)
        return self._construct_counterfactual(true_joint_sig, agent_idx)
    
    def _construct_null_signature(self,
                                 episode_trajectories: List[List] = None,
                                 encoder=None,
                                 config=None) -> np.ndarray:
        """Construct Z-bar: the joint signature with ALL agents replaced by baseline."""
        total_dim = self.n_agents * self.per_agent_dim

        if self.baseline_mode == 'zero_padding':
            return np.zeros(total_dim)

        if self.baseline_mode == 'mean_loss':
            if self._loss_mean is not None:
                return np.tile(self._loss_mean, self.n_agents)
            return np.zeros(total_dim)

        if self.baseline_mode == 'mean_all':
            if self._all_mean is not None:
                return np.tile(self._all_mean, self.n_agents)
            return np.zeros(total_dim)

        if self.baseline_mode == 'medoid':
            if self._medoid_loss is not None:
                if (episode_trajectories is not None and encoder is not None
                        and config is not None and self._loss_trajs is not None
                        and self._medoid_idx is not None):
                    T_win = len(episode_trajectories[0])
                    medoid_traj = self._loss_trajs[self._medoid_idx]
                    null_parts = []
                    for i in range(self.n_agents):
                        truncated = medoid_traj[i][:T_win]
                        null_parts.append(encoder._create_behavioral_signature(truncated, i, config))
                    return np.concatenate(null_parts)
                return self._medoid_loss.copy()
            return np.zeros(total_dim)

        if self.baseline_mode in ('random_valid', 'multi_sample'):
            if self._X_loss is not None and len(self._X_loss) > 0:
                rand_idx = np.random.choice(len(self._X_loss))
                return self._X_loss[rand_idx].copy()
            return np.zeros(total_dim)

        return np.zeros(total_dim)

    def compute_reward(self, joint_signature: np.ndarray,
                       agent_idx: int,
                       reducer: BaseDimensionalityReducer,
                       density_model: BaseDensityEstimator,
                       episode_trajectories: List[List] = None,
                       encoder = None,
                       config = None,
                       agent_signatures: List[np.ndarray] = None) -> float:
        """
        Compute R_int(i) = F(Z) - F(Z^{-i})
        """
        if self.baseline_mode == 'multi_sample' and self._X_loss is not None and len(self._X_loss) > 0:
            n_avail = len(self._X_loss)
            K = min(self.n_samples, n_avail)
            replace = K > n_avail
            rand_indices = np.random.choice(n_avail, size=K, replace=replace)
            cf_sigs = []
            for k in range(K):
                l_idx = rand_indices[k]
                if episode_trajectories is not None and encoder is not None and config is not None:
                    cf_sig = self._construct_counterfactual_signature_for_loss_idx(
                        episode_trajectories, agent_idx, l_idx, encoder, config, agent_signatures=agent_signatures
                    )
                else:
                    start = agent_idx * self.per_agent_dim
                    end = start + self.per_agent_dim
                    cf_sig = joint_signature.copy()
                    cf_sig[start:end] = self._X_loss[l_idx][start:end]
                cf_sigs.append(cf_sig)
            
            joint_sig_batch = joint_signature.reshape(1, -1)
            cf_sigs_batch = np.vstack(cf_sigs)
            all_sigs = np.vstack([joint_sig_batch, cf_sigs_batch])
            
            Z_all = reducer.transform(all_sigs)
            scores = density_model.score(Z_all)
            
            f_z = scores[0]
            cf_scores = scores[1:]
            
            reward = f_z - np.mean(cf_scores)
            
            # Track running max absolute value for reward normalization
            if self.running_max_abs is None:
                self.running_max_abs = max(1.0, abs(reward))
            else:
                self.running_max_abs = max(1.0, 0.99 * self.running_max_abs + 0.01 * abs(reward))
            normalized = reward / self.running_max_abs
            if self.clip_val is not None:
                normalized = np.clip(normalized, -self.clip_val, self.clip_val)
            return normalized

        if episode_trajectories is not None and encoder is not None and config is not None:
            cf_sig = self._construct_counterfactual_signature(
                episode_trajectories, agent_idx, encoder, config, agent_signatures=agent_signatures
            )
        else:
            cf_sig = self._construct_counterfactual(joint_signature, agent_idx)
            
        joint_sig_batch = joint_signature.reshape(1, -1)
        cf_sig_batch = cf_sig.reshape(1, -1)
        all_sigs = np.vstack([joint_sig_batch, cf_sig_batch])
        
        Z_all = reducer.transform(all_sigs)
        scores = density_model.score(Z_all)
        
        f_z = scores[0]
        f_z_cf = scores[1]
        reward = f_z - f_z_cf
        
        # Track running max absolute value for reward normalization
        if self.running_max_abs is None:
            self.running_max_abs = max(1.0, abs(reward))
        else:
            self.running_max_abs = max(1.0, 0.99 * self.running_max_abs + 0.01 * abs(reward))
        
        # Scale/normalize reward
        normalized_reward = reward / max(1e-8, self.running_max_abs)
        if self.clip_val is not None:
            normalized_reward = np.clip(normalized_reward, -self.clip_val, self.clip_val)
        return normalized_reward
    
    def compute_all_rewards(self, joint_signature: np.ndarray,
                             reducer: BaseDimensionalityReducer, 
                             density_model: BaseDensityEstimator,
                             episode_trajectories: List[List] = None,
                             encoder = None,
                             config = None,
                             agent_signatures: List[np.ndarray] = None,
                             return_raw_scores: bool = False) -> np.ndarray:
        """Compute R_int for all agents. Returns normalized array of shape (n_agents,)."""
        # Populated on every call so opt-in forensic logging can inspect the
        # exact donors and scores used by training without drawing any extra
        # random samples.  This attribute is observational; reward calculation
        # and the public return signature remain unchanged.
        self.last_reward_diagnostics = None
        if self.baseline_mode == 'multi_sample' and self._X_loss is not None and len(self._X_loss) > 0:
            n_avail = len(self._X_loss)
            K = min(self.n_samples, n_avail)
            replace = K > n_avail
            # Sample K loss indices ONCE so all agents share the exact same set of reference loss episodes
            rand_indices = np.random.choice(n_avail, size=K, replace=replace)
            
            all_cf_sigs = []
            for i in range(self.n_agents):
                for k in range(K):
                    l_idx = rand_indices[k]
                    if episode_trajectories is not None and encoder is not None and config is not None:
                        cf_sig = self._construct_counterfactual_signature_for_loss_idx(
                            episode_trajectories, i, l_idx, encoder, config, agent_signatures=agent_signatures
                        )
                    else:
                        start = i * self.per_agent_dim
                        end = start + self.per_agent_dim
                        cf_sig = joint_signature.copy()
                        cf_sig[start:end] = self._X_loss[l_idx][start:end]
                    all_cf_sigs.append(cf_sig)
            
            joint_sig_batch = joint_signature.reshape(1, -1)
            cf_sigs_batch = np.vstack(all_cf_sigs)
            all_sigs = np.vstack([joint_sig_batch, cf_sigs_batch])
            
            Z_all = reducer.transform(all_sigs)
            scores = density_model.score(Z_all)
            
            f_z = scores[0]
            cf_scores = scores[1:]  # shape: (n_agents * K,)
            
            rewards = np.zeros(self.n_agents, dtype=np.float32)
            f_z_cfs = np.zeros(self.n_agents, dtype=np.float32)
            for i in range(self.n_agents):
                avg_f_cf_i = np.mean(cf_scores[i * K : (i + 1) * K])
                f_z_cfs[i] = avg_f_cf_i
                rewards[i] = f_z - avg_f_cf_i
            raw_rewards = rewards.copy()
            max_abs = np.max(np.abs(rewards))
            running_scale_before = self.running_max_abs
            if self.running_max_abs is None:
                self.running_max_abs = max(1.0, max_abs)
            else:
                # self.running_max_abs = max(1.0, 0.99 * self.running_max_abs + 0.01 * max_abs)
                self.running_max_abs =  0.99 * self.running_max_abs + 0.01 * max_abs
            normalized_rewards = rewards / max(1e-8, self.running_max_abs)
            if self.clip_val is not None:
                normalized_rewards = np.clip(normalized_rewards, -self.clip_val, self.clip_val)

            self.last_reward_diagnostics = {
                "sampled_donor_indices": rand_indices.astype(int).tolist(),
                "sampled_cf_scores": [
                    cf_scores[i * K : (i + 1) * K].astype(float).tolist()
                    for i in range(self.n_agents)
                ],
                "sampled_cf_signatures": [
                    np.asarray(all_cf_sigs[i * K : (i + 1) * K], dtype=float)
                    for i in range(self.n_agents)
                ],
                "running_scale_before": running_scale_before,
                "running_scale_after": float(self.running_max_abs),
                "unclipped_normalized_rewards": (
                    raw_rewards / max(1e-8, self.running_max_abs)
                ).astype(float).tolist(),
            }

            if return_raw_scores:
                return normalized_rewards, f_z, f_z_cfs, raw_rewards
            return normalized_rewards

        cf_sigs = []
        for i in range(self.n_agents):
            if episode_trajectories is not None and encoder is not None and config is not None:
                cf_sig = self._construct_counterfactual_signature(
                    episode_trajectories, i, encoder, config, agent_signatures=agent_signatures
                )
            else:
                cf_sig = self._construct_counterfactual(joint_signature, i)
            cf_sigs.append(cf_sig)
            
        joint_sig_batch = joint_signature.reshape(1, -1)
        cf_sigs_batch = np.vstack(cf_sigs)
        all_sigs = np.vstack([joint_sig_batch, cf_sigs_batch])
        
        Z_all = reducer.transform(all_sigs)
        scores = density_model.score(Z_all)
        
        f_z = scores[0]
        f_z_cfs = scores[1:]
        rewards = f_z - f_z_cfs
        raw_rewards = rewards.copy()
        
        # Track running max absolute value for reward normalization
        max_abs = np.max(np.abs(rewards))
        running_scale_before = self.running_max_abs
        if self.running_max_abs is None:
            self.running_max_abs = max(1.0, max_abs)
        else:
            self.running_max_abs = max(1.0, 0.99 * self.running_max_abs + 0.01 * max_abs)

        # Scale/normalize rewards
        normalized_rewards = rewards / max(1e-8, self.running_max_abs)
        if self.clip_val is not None:
            normalized_rewards = np.clip(normalized_rewards, -self.clip_val, self.clip_val)

        self.last_reward_diagnostics = {
            "sampled_donor_indices": [],
            "sampled_cf_scores": [[float(f_z_cfs[i])] for i in range(self.n_agents)],
            "sampled_cf_signatures": [
                np.asarray([cf_sigs[i]], dtype=float) for i in range(self.n_agents)
            ],
            "running_scale_before": running_scale_before,
            "running_scale_after": float(self.running_max_abs),
            "unclipped_normalized_rewards": (
                raw_rewards / max(1e-8, self.running_max_abs)
            ).astype(float).tolist(),
        }
            
        if return_raw_scores:
            return normalized_rewards, f_z, f_z_cfs, raw_rewards
        return normalized_rewards

    def compute_synergy_diagnostic(self,
                                   joint_signature: np.ndarray,
                                   reducer: BaseDimensionalityReducer,
                                   density_model: BaseDensityEstimator,
                                   f_z: float = None,
                                   f_z_cfs: np.ndarray = None,
                                   episode_trajectories: List[List] = None,
                                   encoder=None,
                                   config=None) -> dict:
        """Compute the Harsanyi two-agent synergy dividend delta_{1,2}.

        Returns {"delta_12": float, "f_z_null": float}.
        Accepts pre-computed f_z and f_z_cfs from compute_all_rewards(return_raw_scores=True).
        """
        if f_z is None or f_z_cfs is None:
            _, f_z, f_z_cfs, _ = self.compute_all_rewards(
                joint_signature, reducer, density_model,
                episode_trajectories=episode_trajectories,
                encoder=encoder, config=config,
                return_raw_scores=True)

        if self.baseline_mode == 'multi_sample' and self._X_loss is not None and len(self._X_loss) > 0:
            K = min(self.n_samples, len(self._X_loss))
            null_sigs = []
            for _ in range(K):
                null_sigs.append(self._construct_null_signature(
                    episode_trajectories, encoder, config))
            null_batch = np.vstack(null_sigs)
            Z_null = reducer.transform(null_batch)
            f_z_null = float(np.mean(density_model.score(Z_null)))
        else:
            null_sig = self._construct_null_signature(
                episode_trajectories, encoder, config)
            Z_null = reducer.transform(null_sig.reshape(1, -1))
            f_z_null = float(density_model.score(Z_null)[0])

        delta_12 = float(f_z) - float(f_z_cfs[0]) - float(f_z_cfs[1]) + f_z_null
        return {"delta_12": delta_12, "f_z_null": f_z_null}

    def spread_reward(self, step_rewards: np.ndarray, r_int_episode: float,
                      mode: str = 'last_step', dynamic_beta: float = 1.0,
                      temporal_weights=None) -> np.ndarray:
        """
        Inject episode-level R_int into per-step rewards.

        Args:
            step_rewards: np.array of shape (T,) — per-step environmental rewards
            r_int_episode: float — episode-level intrinsic reward for this agent
            mode: 'last_step' (default), 'uniform', or 'gae_magnitude'
            dynamic_beta: float — current beta schedule value
            temporal_weights: normalized factual teacher |GAE| weights for gae_magnitude

        Returns:
            np.array of shape (T,) — modified rewards
        """
        shaped = np.array(step_rewards, dtype=np.float32).copy()
        scaled_r_int = dynamic_beta * r_int_episode

        if mode == 'last_step':
            if len(shaped) > 0:
                shaped[-1] += scaled_r_int
        elif mode == 'uniform':
            if len(shaped) > 0:
                shaped += scaled_r_int / len(shaped)
        elif mode == 'gae_magnitude':
            weights = np.asarray(temporal_weights, dtype=np.float64)
            if (weights.shape != shaped.shape or not np.isfinite(weights).all()
                    or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0)):
                raise ValueError('gae_magnitude requires finite nonnegative temporal weights summing to one')
            shaped += scaled_r_int * weights

        return shaped
