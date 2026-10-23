import numpy as np
from abc import ABC, abstractmethod
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

class BaseDensityEstimator(ABC):
    """
    Abstract base class for density estimators used in counterfactual reward shaping.
    
    The core idea: We want to measure how much a joint behavioral signature Z
    "belongs" to the distribution of successful (winning) strategies vs failed
    (losing) strategies.
    
    Mathematically, each estimator computes a scoring function F(Z) that represents:
        F(Z) = measure_of_similarity(Z, winning_distribution) - measure_of_similarity(Z, losing_distribution)
    
    A positive F(Z) means Z looks more like a winning strategy.
    A negative F(Z) means Z looks more like a losing strategy.
    
    This score is then used in the counterfactual reward computation:
        R_int(i) = F(Z) - F(Z^{-i})
    where Z^{-i} is the joint signature with agent i's contribution replaced by a baseline.
    
    Different subclasses implement different density estimation approaches:
    - CentroidDensityEstimator: Simple distance to cluster centers (fast, assumes unimodal)
    - GMMDensityEstimator: Gaussian Mixture Models (handles multi-modal strategies like Tank/Healer vs Double Attacker)
    - KDEDensityEstimator: Kernel Density Estimation (non-parametric, captures complex geometric boundaries)
    - NormalizingFlowEstimator: Deep generative model (placeholder, for complex topologies)
    """
    
    @abstractmethod
    def fit(self, Z_win: np.ndarray, Z_loss: np.ndarray) -> 'BaseDensityEstimator':
        """
        Fit the density model on winning and losing latent-space representations.
        
        Args:
            Z_win: Array of shape (n_win_episodes, latent_dim) — winning strategy embeddings
            Z_loss: Array of shape (n_loss_episodes, latent_dim) — losing strategy embeddings
        """
        pass
    
    @abstractmethod
    def score(self, Z: np.ndarray) -> np.ndarray:
        """
        Score joint behavioral signatures.
        
        Returns F(Z) = similarity_to_win - similarity_to_loss.
        Positive means Z looks like a winning strategy.
        
        Args:
            Z: Array of shape (n_samples, latent_dim)
        Returns:
            np.ndarray: The density-based scores
        """
        pass


class CentroidDensityEstimator(BaseDensityEstimator):
    def __init__(self):
        self.win_centroid = None
        self.loss_centroid = None
        
    def fit(self, Z_win: np.ndarray, Z_loss: np.ndarray) -> 'CentroidDensityEstimator':
        self.win_centroid = np.mean(Z_win, axis=0)
        self.loss_centroid = np.mean(Z_loss, axis=0)
        return self
        
    def score(self, Z: np.ndarray) -> np.ndarray:
        """
        Positive when Z is closer to winning centroid. Assumes unimodal, spherical clusters.
        """
        dist_to_win = np.sum((Z - self.win_centroid)**2, axis=1)
        dist_to_loss = np.sum((Z - self.loss_centroid)**2, axis=1)
        return dist_to_loss - dist_to_win


class GMMDensityEstimator(BaseDensityEstimator):
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.gmm_win = None
        self.gmm_loss = None
        
    def fit(self, Z_win: np.ndarray, Z_loss: np.ndarray) -> 'GMMDensityEstimator':
        n_comp_win = min(self.n_components, len(Z_win))
        n_comp_loss = min(self.n_components, len(Z_loss))
        
        self.gmm_win = GaussianMixture(n_components=n_comp_win)
        self.gmm_win.fit(Z_win)
        
        self.gmm_loss = GaussianMixture(n_components=n_comp_loss)
        self.gmm_loss.fit(Z_loss)
        return self
        
    def score(self, Z: np.ndarray) -> np.ndarray:
        """
        GMM captures multiple strategy clusters. 
        Returns log-likelihood difference: log P(Z | win) - log P(Z | loss).
        """
        score_win = self.gmm_win.score_samples(Z)
        score_loss = self.gmm_loss.score_samples(Z)
        return score_win - score_loss


class KDEDensityEstimator(BaseDensityEstimator):
    """
    Fits Kernel Density Estimation on latents.
    """
    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = bandwidth
        self.win_kde = None
        self.loss_kde = None
        
    def fit(self, win_buffer: np.ndarray, loss_buffer: np.ndarray):
        from sklearn.neighbors import KernelDensity
        if len(win_buffer) > 0:
            self.win_kde = KernelDensity(bandwidth=self.bandwidth).fit(win_buffer)
        if len(loss_buffer) > 0:
            self.loss_kde = KernelDensity(bandwidth=self.bandwidth).fit(loss_buffer)
            
    def score(self, Z: np.ndarray) -> np.ndarray:
        if self.win_kde is None or self.loss_kde is None:
            return np.zeros(len(Z))
            
        win_log_prob = self.win_kde.score_samples(Z)
        loss_log_prob = self.loss_kde.score_samples(Z)
        
        # log(P(win)) - log(P(loss)) = log(P(win)/P(loss))
        return win_log_prob - loss_log_prob

class WassersteinDensityEstimator(BaseDensityEstimator):
    """
    Computes the exact Wasserstein-1 distance between empirical occupancy measures.
    Expects input Z to be the unreduced output of StateActionOccupancyEncoder OR SpatialHeatmapEncoder.
    """
    def __init__(self, grid_size: int, action_dim: int, n_agents: int, action_cost: float = 1.0, is_heatmap: bool = False):
        self.grid_size = grid_size
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.action_cost = action_cost
        self.is_heatmap = is_heatmap
        
        if self.is_heatmap:
            self.per_agent_dim = grid_size * grid_size
        else:
            self.per_agent_dim = grid_size * grid_size * action_dim
            
        self.M = self._build_cost_matrix()
        
        self.win_prototype = None
        self.loss_prototype = None
        
    def _build_cost_matrix(self):
        """Builds the ground cost matrix based on physical geometry."""
        dim = self.per_agent_dim
        M = np.zeros((dim, dim), dtype=np.float64)
        
        if self.is_heatmap:
            for y1 in range(self.grid_size):
                for x1 in range(self.grid_size):
                    idx1 = y1 * self.grid_size + x1
                    for y2 in range(self.grid_size):
                        for x2 in range(self.grid_size):
                            idx2 = y2 * self.grid_size + x2
                            # Manhattan distance
                            M[idx1, idx2] = abs(y1 - y2) + abs(x1 - x2)
        else:
            for y1 in range(self.grid_size):
                for x1 in range(self.grid_size):
                    for a1 in range(self.action_dim):
                        idx1 = y1 * (self.grid_size * self.action_dim) + x1 * self.action_dim + a1
                        
                        for y2 in range(self.grid_size):
                            for x2 in range(self.grid_size):
                                for a2 in range(self.action_dim):
                                    idx2 = y2 * (self.grid_size * self.action_dim) + x2 * self.action_dim + a2
                                    
                                    # Manhattan distance + discrete action cost
                                    dist = abs(y1 - y2) + abs(x1 - x2)
                                    if a1 != a2:
                                        dist += self.action_cost
                                        
                                    M[idx1, idx2] = dist
        return M
        
    def fit(self, win_buffer: np.ndarray, loss_buffer: np.ndarray):
        """
        The arithmetic mean of multiple empirical occupancy measures is exactly
        the empirical occupancy measure of the mixture policy. 
        Thus, we do not need an iterative Wasserstein Barycenter here!
        """
        if len(win_buffer) > 0:
            self.win_prototype = np.mean(win_buffer, axis=0)
        if len(loss_buffer) > 0:
            self.loss_prototype = np.mean(loss_buffer, axis=0)
            
    def score(self, Z: np.ndarray) -> np.ndarray:
        import ot
        
        if self.win_prototype is None or self.loss_prototype is None:
            return np.zeros(len(Z))
            
        batch_size = len(Z)
        scores = np.zeros(batch_size)
        
        win_proto_agents = self.win_prototype.reshape(self.n_agents, self.per_agent_dim)
        loss_proto_agents = self.loss_prototype.reshape(self.n_agents, self.per_agent_dim)
        Z_agents = Z.reshape(batch_size, self.n_agents, self.per_agent_dim)
        
        for b in range(batch_size):
            b_score = 0.0
            for i in range(self.n_agents):
                p = np.asarray(Z_agents[b, i], dtype=np.float64)
                q_win = np.asarray(win_proto_agents[i], dtype=np.float64)
                q_loss = np.asarray(loss_proto_agents[i], dtype=np.float64)
                
                if p.sum() == 0 or q_win.sum() == 0 or q_loss.sum() == 0:
                    continue
                    
                p = p / p.sum()
                q_win = q_win / q_win.sum()
                q_loss = q_loss / q_loss.sum()
                
                w_win = ot.emd2(p, q_win, self.M)
                w_loss = ot.emd2(p, q_loss, self.M)
                
                # We want to be closer to win than loss. 
                # Distance is smaller when closer. 
                # Score = w_loss - w_win (High score means close to win)
                b_score += (w_loss - w_win)
                
            scores[b] = b_score
            
        return scores


class PassthroughDensityEstimator(BaseDensityEstimator):
    """
    Identity scorer for use with MixerReducer.
    The mixer's transform() already outputs scalar scores,
    so this just passes them through.
    """
    def fit(self, Z_win: np.ndarray, Z_loss: np.ndarray):
        # No fitting needed — the mixer handles everything
        return self
    
    def score(self, Z: np.ndarray) -> np.ndarray:
        # Z shape: (batch, 1) from MixerReducer.transform()
        return Z.squeeze(-1)  # -> (batch,)


class NormalizingFlowEstimator(BaseDensityEstimator):
    """
    Placeholder for Deep Generative Model (RealNVP) to map latent strategy
    distributions to a simple normal distribution.
    """
    def fit(self, Z_win: np.ndarray, Z_loss: np.ndarray) -> 'NormalizingFlowEstimator':
        raise NotImplementedError("NormalizingFlowEstimator is not yet implemented")
        
    def score(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("NormalizingFlowEstimator is not yet implemented")
