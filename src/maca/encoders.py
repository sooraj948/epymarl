import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from .config import MACAConfig

class BaseTrajectoryEncoder(ABC):
    """
    Abstract base class for encoding raw trajectory data into fixed-length joint signatures.
    """
    @abstractmethod
    def encode(self, episode_trajectories: List[List[Tuple]], n_agents: int, config: MACAConfig) -> np.ndarray:
        """
        Encodes the episode trajectories into a single fixed-length joint signature.
        
        Args:
            episode_trajectories: List of per-agent trajectory data [agent_i][(obs, action), ...]
            n_agents: Number of agents
            config: Configuration object
            
        Returns:
            np.ndarray: The encoded joint signature (1D array)
        """
        pass
        
    @property
    @abstractmethod
    def per_agent_dim(self) -> int:
        pass
        
    def joint_dim(self, n_agents: int) -> int:
        return self.per_agent_dim * n_agents


class SpatialHeatmapEncoder(BaseTrajectoryEncoder):
    """
    Creates a spatial heatmap of state visitations.
    """
    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        
    @property
    def per_agent_dim(self) -> int:
        return self.grid_size * self.grid_size
        
    def _create_behavioral_signature(self, trajectory: List[Tuple], agent_id: int, config: MACAConfig) -> np.ndarray:
        """
        Creates a normalized visitation heatmap for a single agent.
        """
        heatmap = np.zeros((self.grid_size, self.grid_size))
        for obs, action in trajectory:
            # In LBF, self is always first in the player portion of the flat observation
            # Player positions start after food. Food is (y, x, level) for each food.
            agent_y_idx = config.n_food * 3
            agent_x_idx = config.n_food * 3 + 1
            
            x = int(obs[agent_x_idx])
            y = int(obs[agent_y_idx])
            
            if x >= 0 and y >= 0:
                heatmap[y, x] += 1
                
        if np.sum(heatmap) > 0:
            heatmap = heatmap / np.sum(heatmap)
            
        return heatmap.flatten()
        
    def encode_per_agent(self, episode_trajectories: List[List[Tuple]], n_agents: int, config: MACAConfig) -> List[np.ndarray]:
        """
        Encodes each agent's trajectory into an individual signature.
        
        Returns:
            List of np.ndarray: Per-agent signature vectors.
        """
        return [self._create_behavioral_signature(episode_trajectories[i], i, config)
                for i in range(n_agents)]
    
    def encode(self, episode_trajectories: List[List[Tuple]], n_agents: int, config: MACAConfig) -> np.ndarray:
        signatures = self.encode_per_agent(episode_trajectories, n_agents, config)
        return np.concatenate(signatures)


class RelativeInteractionEncoder(BaseTrajectoryEncoder):
    """
    Computes per-timestep relational features (inter-agent dist, agent-to-food dist)
    and aggregates them across time using summary statistics (mean, std, min, max).
    """
    def __init__(self, n_food: int):
        self.n_food = n_food
        self._per_agent_dim = 2 * 4 # 2 features * 4 stats
        
    @property
    def per_agent_dim(self) -> int:
        return self._per_agent_dim
        
    def encode(self, episode_trajectories: List[List[Tuple]], n_agents: int, config: MACAConfig) -> np.ndarray:
        signatures = []
        for i in range(n_agents):
            traj = episode_trajectories[i]
            if len(traj) == 0:
                signatures.append(np.zeros(self.per_agent_dim))
                continue
                
            agent_feats = []
            for obs, action in traj:
                # self position (always first player in the player portion of obs)
                self_y = obs[config.n_food * 3]
                self_x = obs[config.n_food * 3 + 1]
                
                # distance to other agents
                other_dists = []
                for other_idx in range(1, config.n_agents):
                    start_idx = config.n_food * 3 + other_idx * 3
                    other_y = obs[start_idx]
                    other_x = obs[start_idx + 1]
                    dist = np.sqrt((self_y - other_y)**2 + (self_x - other_x)**2)
                    other_dists.append(dist)
                
                min_agent_dist = min(other_dists) if other_dists else 0.0
                
                # distance to foods
                food_dists = []
                for f_idx in range(config.n_food):
                    food_y = obs[f_idx * 3]
                    food_x = obs[f_idx * 3 + 1]
                    # Check if food is active (non-negative coordinates)
                    if food_y >= 0 and food_x >= 0:
                        dist = np.sqrt((self_y - food_y)**2 + (self_x - food_x)**2)
                        food_dists.append(dist)
                        
                min_food_dist = min(food_dists) if food_dists else 0.0
                
                agent_feats.append([min_agent_dist, min_food_dist])
                
            agent_feats = np.array(agent_feats)
            mean = np.mean(agent_feats, axis=0)
            std = np.std(agent_feats, axis=0)
            min_val = np.min(agent_feats, axis=0)
            max_val = np.max(agent_feats, axis=0)
            
            agent_sig = np.concatenate([mean, std, min_val, max_val])
            signatures.append(agent_sig)
            
        return np.concatenate(signatures)


class GraphAdjacencyEncoder(BaseTrajectoryEncoder):
    """
    Models episode as a fully-connected graph where nodes are agents and objects.
    Edges represent average proximity over the episode.
    Outputs the flattened adjacency matrix.
    """
    @property
    def per_agent_dim(self) -> int:
        return 16 
        
    def encode(self, episode_trajectories: List[List[Tuple]], n_agents: int, config: MACAConfig) -> np.ndarray:
        raise NotImplementedError("GraphAdjacencyEncoder is not yet implemented")


class StateActionOccupancyEncoder(BaseTrajectoryEncoder):
    """
    Creates a state-action occupancy measure (visitation heatmap across states and actions).
    """
    def __init__(self, grid_size: int, action_dim: int):
        self.grid_size = grid_size
        self.action_dim = action_dim
        
    @property
    def per_agent_dim(self) -> int:
        return self.grid_size * self.grid_size * self.action_dim
        
    def _create_behavioral_signature(self, trajectory: List[Tuple], agent_id: int, config: MACAConfig) -> np.ndarray:
        """
        Creates a normalized state-action visitation heatmap for a single agent.
        """
        occupancy = np.zeros((self.grid_size, self.grid_size, self.action_dim), dtype=np.float32)
        for obs, action in trajectory:
            agent_y_idx = config.n_food * 3
            agent_x_idx = config.n_food * 3 + 1
            
            x = int(obs[agent_x_idx])
            y = int(obs[agent_y_idx])
            
            if isinstance(action, (np.ndarray, list)):
                a = int(action[0])
            else:
                a = int(action)
                
            if x >= 0 and y >= 0 and 0 <= a < self.action_dim:
                occupancy[y, x, a] += 1.0
                
        if np.sum(occupancy) > 0:
            occupancy = occupancy / np.sum(occupancy)
            
        return occupancy.flatten()
        
    def encode_per_agent(self, episode_trajectories: List[List[Tuple]], n_agents: int, config: MACAConfig) -> List[np.ndarray]:
        """
        Encodes each agent's trajectory into an individual signature.
        """
        return [self._create_behavioral_signature(episode_trajectories[i], i, config)
                for i in range(n_agents)]
    
    def encode(self, episode_trajectories: List[List[Tuple]], n_agents: int, config: MACAConfig) -> np.ndarray:
        signatures = self.encode_per_agent(episode_trajectories, n_agents, config)
        return np.concatenate(signatures)


class TerminalGoalEncoder:
    """Per-agent terminal position followed by factual remaining-food coordinates.

    Each food uses two slots. Eaten/absent food is (-1, -1), so the vector
    stays at 2 + 2*n_food entries without a separate presence bit.
    """

    def __init__(self, n_food: int):
        self.n_food = n_food

    @property
    def per_agent_dim(self) -> int:
        return 2 + 2 * self.n_food

    def joint_dim(self, n_agents: int) -> int:
        return self.per_agent_dim * n_agents

    def encode_per_agent_from_state(self, state, config) -> List[np.ndarray]:
        foods = np.asarray(state["food_grid"])
        coords = np.argwhere(foods > 0)
        if len(coords) > self.n_food:
            raise ValueError("State contains more foods than the encoder supports")
        scale = float(config.grid_size - 1)
        food_slots = np.full((self.n_food, 2), -1.0, dtype=np.float32)
        food_slots[:len(coords)] = coords / scale
        context = food_slots.ravel()
        return [np.concatenate((np.asarray(player["position"], dtype=np.float32) / scale,
                                context)).astype(np.float32)
                for player in state["players"]]

    def encode_from_state(self, state, config) -> np.ndarray:
        return np.concatenate(self.encode_per_agent_from_state(state, config))
