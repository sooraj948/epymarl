import numpy as np
from collections import deque
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class EpisodeRecord:
    """Stores both raw trajectory data and encoded signature for a single episode."""
    raw_trajectories: List[List]  # raw per-agent trajectory data: [agent_i][(obs, action), ...]
    joint_signature: np.ndarray   # pre-encoded joint behavioral vector
    env_reward: float             # team reward for this episode

class JointEpisodeBuffer:
    """
    Maintains separate winning and losing repositories of joint trajectories.
    
    From the plan:
    - Winning buffer (B_win): episodes where env_reward > 0
    - Losing buffer (B_loss): episodes where env_reward <= 0  (NOTE: in LBF, 0 reward = failure)
    - FIFO sliding window for eviction
    - Active balancing sampler for 50/50 blend
    """
    
    def __init__(self, buffer_capacity: int = 500, 
                 n_min_win: int = 20, n_min_loss: int = 20,
                 use_jitter: bool = False):
        self.win_buffer: deque[EpisodeRecord] = deque(maxlen=buffer_capacity)
        self.loss_buffer: deque[EpisodeRecord] = deque(maxlen=buffer_capacity)
        self.n_min_win = n_min_win
        self.n_min_loss = n_min_loss
        self.use_jitter = use_jitter
    
    def store(self, raw_trajectories: List[List], 
              joint_signature: np.ndarray, env_reward: float):
        """Route episode to win or loss buffer based on reward."""
        record = EpisodeRecord(raw_trajectories, joint_signature, env_reward)
        if env_reward > 0:
            self.win_buffer.append(record)
        else:
            self.loss_buffer.append(record)
    
    @property
    def is_ready(self) -> bool:
        """True when both buffers have enough data for stable model fitting."""
        return (len(self.win_buffer) >= self.n_min_win and 
                len(self.loss_buffer) >= self.n_min_loss)
    
    def sample_balanced_batch(self, batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (X_win, X_loss) with equal counts.
        
        If batch_size is None, uses all available data with the minority
        class upsampled to match the majority class.
        If batch_size is given, samples batch_size//2 from each buffer
        (with replacement if needed for the minority class).
        """
        # Get all signatures
        win_sigs = np.array([r.joint_signature for r in self.win_buffer])
        loss_sigs = np.array([r.joint_signature for r in self.loss_buffer])
        
        if batch_size is None:
            # Use all data, upsample minority
            n_target = max(len(win_sigs), len(loss_sigs))
        else:
            n_target = batch_size // 2
        
        # Upsample if needed
        if len(win_sigs) < n_target:
            indices = np.random.choice(len(win_sigs), n_target, replace=True)
            win_sigs = win_sigs[indices]
            if self.use_jitter:
                # Jittering: Add small Gaussian noise to the upsampled signatures to augment the dataset
                jitter = np.random.normal(loc=0.0, scale=0.02, size=win_sigs.shape)
                win_sigs = win_sigs + jitter
                win_sigs = np.clip(win_sigs, 0.0, 1.0)
        elif len(win_sigs) > n_target:
            indices = np.random.choice(len(win_sigs), n_target, replace=False)
            win_sigs = win_sigs[indices]
            
        if len(loss_sigs) < n_target:
            indices = np.random.choice(len(loss_sigs), n_target, replace=True)
            loss_sigs = loss_sigs[indices]
        elif len(loss_sigs) > n_target:
            indices = np.random.choice(len(loss_sigs), n_target, replace=False)
            loss_sigs = loss_sigs[indices]
        
        return win_sigs, loss_sigs
    
    def get_all_signatures(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X_win, X_loss) without balancing."""
        win_sigs = np.array([r.joint_signature for r in self.win_buffer])
        loss_sigs = np.array([r.joint_signature for r in self.loss_buffer])
        return win_sigs, loss_sigs

    def get_all_signatures_and_returns(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return stored signatures and their original, graded returns.

        No class balancing or minority upsampling is applied. This preserves
        the value-regression target instead of changing it into a balanced
        win/loss objective. Records are ordered deterministically as wins then
        losses; DataLoader shuffling controls optimization order downstream.
        """
        records = list(self.win_buffer) + list(self.loss_buffer)
        signatures = np.array([record.joint_signature for record in records])
        returns = np.array([record.env_reward for record in records], dtype=np.float32)
        return signatures, returns
    
    def get_all_raw_trajectories(self) -> Tuple[List, List]:
        """Returns (win_trajs, loss_trajs) as raw trajectory lists."""
        win_trajs = [r.raw_trajectories for r in self.win_buffer]
        loss_trajs = [r.raw_trajectories for r in self.loss_buffer]
        return win_trajs, loss_trajs
    
    def __len__(self):
        return len(self.win_buffer) + len(self.loss_buffer)
    
    def stats(self) -> dict:
        return {
            'n_wins': len(self.win_buffer),
            'n_losses': len(self.loss_buffer),
            'is_ready': self.is_ready,
            'total': len(self)
        }
