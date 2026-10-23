from dataclasses import dataclass

@dataclass
class MACAConfig:
    # Environment
    n_agents: int = 2
    n_food: int = 2
    grid_size: int = 5
    max_ep_len: int = 200
    env_name: str = "Foraging-5x5-2p-2f-coop-v3"
    
    # Training
    n_episodes: int = 12000
    lr: float = 0.001
    gamma: float = 0.99
    seed: int = 67
    batch_size: int = 16  # episodes per policy update
    
    # Buffer
    buffer_capacity: int = 500  # per win/loss buffer
    n_min_win: int = 100
    n_min_loss: int = 20
    fit_freq: int = 100  # how often to refit models
    warmup_episodes: int = 4000
    
    # MACA reward shaping
    beta: float = 1.0  # intrinsic reward weight
    beta_mode: str = "dynamic" # options: dynamic, fixed, normalized_adv, critic_confidence
    beta_saturation_c: float = 1.0 # saturation constant c for critic_confidence gating
    running_stat_decay: float = 0.01 # decay rate for EMA in running stats (std, var)
    adv_tracking_mode: str = "shaped" # options: shaped, unshaped (for normalized_adv)
    baseline_mode: str = "medoid"  # options: zero_padding, mean_loss, mean_all, medoid, random_valid, multi_sample
    n_samples: int = 16  # number of loss samples to average for multi_sample baseline
    n_components: int = 5  # latent space dimensionality
    clip_rewards: bool = False # whether to clip intrinsic rewards
    clip_val: float = 1.0  # clipping boundary for intrinsic rewards (only active if clip_rewards=True)
    beta_threshold: float = 0.3  # running average return threshold for dynamic beta scheduling
    beta_ramp: float = 0.35  # running average return ramp-up interval for dynamic beta scheduling
    
    # Module selections
    encoder_type: str = "heatmap"  # options: heatmap, occupancy, interaction, graph
    reducer_type: str = "pca"  # options: pca, umap, contrastive, none, mixer, trajectory_return, mixer_monotonic
    density_type: str = "centroid"  # options: centroid, gmm, kde, wasserstein
    terminal_goal_input: bool = False  # Pipeline 2: terminal positions and remaining-food coordinates
    occupancy_goal_input: bool = False  # Pipelines 2.5/5.5: append initial and terminal food coordinates
    average_pbrs_score: bool = False  # Pipelines 1.5/2.5: mean cooperative potential
    mean_advantage_score: bool = False  # Pipelines 3.5/5.5: mean critic TD residual
    gae_advantage_score: bool = False  # Pipelines 3.5/5.5: mean GAE using PPO lambda
    score_std_scaling: bool = False  # Divide scores by previous factual rollout batch's GAE std
    oracle_cf_diagnostic: bool = False  # Pipeline 5.5: read-only reset-oracle comparison
    occupancy_cf_diagnostic: bool = False  # Add splice/oracle occupancy distances and baseline
    occupancy_mean_cf_diagnostic: bool = False  # Compare 16-splice and 8-oracle mean occupancies
    reset_oracle_mixer_cf: bool = False  # Pipeline 5.5.1: score native reset CF occupancies with the mixer
    action_replay_mixer_cf: bool = False  # Pipeline 5.5.2: replay recorded teammate actions from observed board
    action_replay_direct_score: bool = False  # Apply teacher credit; retain shadow mixer for matched controls
    online_critic_score: bool = False  # Pipeline 3.5: score with current MAPPO critic
    negate_mean_advantage_score: bool = False  # Pipeline 3.5 sign-flip ablation
    match_mixer_reward_schedule: bool = False  # Exact scorer with Pipeline 2 beta/injection
    direct_injection: str = "last_step"  # Exact scorer reward placement without beta schedule
    
    # GMM specific
    gmm_n_components: int = 3
    
    # KDE specific
    kde_bandwidth: float = 1.0
    
    # Wasserstein specific
    wasserstein_action_cost: float = 1.0
    
    # UMAP specific
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    
    # Contrastive encoder specific
    contrastive_lr: float = 0.001
    contrastive_epochs: int = 20
    contrastive_batch_size: int = 64
    contrastive_temperature: float = 0.1

    # Neural trajectory scorer settings (BCE mixers and return regressor)
    mixer_lr: float = 0.001
    mixer_epochs: int = 20
    mixer_batch_size: int = 64
    mixer_var_reg: float = 0.01   # variance regularization coefficient (collapse prevention)
    mixer_embed_dim: int = 32     # only used by mixer_monotonic
    mixer_target_scale: float = 1.0  # diagnostic: train scalar mixer in scaled units, return original units
    
    # Observation dimensions (computed from env)
    obs_dim_flat: int = 12  # (N_FOOD * 3) + (N_AGENTS * 3) for LBF
    action_dim: int = 6  # LBF has 6 actions

    # Algorithm & MAPPO / PPO Hyperparameters
    algorithm: str = "reinforce"        # options: reinforce, mappo
    ppo_clip_eps: float = 0.2           # PPO clipping epsilon
    ppo_epochs: int = 4                 # Number of optimization epochs per batch
    ppo_value_coef: float = 0.5         # Value loss coefficient
    ppo_gae_lambda: float = 0.95        # GAE lambda
    ppo_max_grad_norm: float = 0.5      # Gradient clipping norm
    log_gradient_norms: bool = False   # Record pre-clipping actor norms for each PPO epoch
    critic_lr: float = 3e-4             # Critic learning rate
    r_int_injection: str = "last_step"  # options: last_step, uniform, gae_magnitude
    centralized_critic: bool = False    # Independent vs Centralized critic
    shared_critic: bool = False         # Share one agent-conditioned joint-observation critic

    # Hand-engineered dense-reward diagnostic
    dense_reward: str = "none"          # options: none, pbrs, raw_distance
    pbrs_alpha: float = 1.0
    raw_distance_alpha: float = 0.1
    terminal_distance_alpha: float = 1.0

    @property
    def behavior_dim(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def beta_ramp_up(self) -> float:
        return self.beta_ramp

    @beta_ramp_up.setter
    def beta_ramp_up(self, value: float):
        self.beta_ramp = value

    @classmethod
    def from_env_name(cls, env_name: str):
        parts = env_name.split('-')
        grid_size = int(parts[1].split('x')[0])
        n_agents = int(parts[2].replace('p', ''))
        n_food = int(parts[3].replace('f', ''))
        obs_dim_flat = (n_food * 3) + (n_agents * 3)
        return cls(
            env_name=env_name,
            grid_size=grid_size,
            n_agents=n_agents,
            n_food=n_food,
            obs_dim_flat=obs_dim_flat
        )
