import numpy as np
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    pass

class BaseDimensionalityReducer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, labels: np.ndarray = None) -> 'BaseDimensionalityReducer':
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def fit_transform(self, X: np.ndarray, labels: np.ndarray = None) -> np.ndarray:
        return self.fit(X, labels).transform(X)


class LinearPCAReducer(BaseDimensionalityReducer):
    def __init__(self, n_components: int = 5):
        self.n_components = n_components
        self.pca = None
        self.scaler = None
        
    def fit(self, X: np.ndarray, labels: np.ndarray = None) -> 'LinearPCAReducer':
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        n_samples, n_features = X_scaled.shape
        n_comp = min(self.n_components, n_samples, n_features)
        self.pca = PCA(n_components=n_comp, svd_solver='randomized')
        self.pca.fit(X_scaled)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA/Scaler is not fitted yet.")
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)


class UMAPReducer(BaseDimensionalityReducer):
    def __init__(self, n_components: int = 5, n_neighbors: int = 15, min_dist: float = 0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.umap_model = None
        
    def fit(self, X: np.ndarray, labels: np.ndarray = None) -> 'UMAPReducer':
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is not installed. Please pip install umap-learn")
            
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist
        )
        self.umap_model.fit(X)
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.umap_model is None:
            raise ValueError("UMAP is not fitted yet.")
        return self.umap_model.transform(X)


class ContrastiveEncoderReducer(BaseDimensionalityReducer):
    def __init__(self, input_dim: int, n_components: int = 5,
                 lr: float = 0.001, epochs: int = 50, batch_size: int = 64, temperature: float = 0.1,
                 seed: int = None):
        self.input_dim = input_dim
        self.n_components = n_components
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.temperature = temperature
        # Dedicated generator for DataLoader shuffling, isolated from
        # torch's global default generator (which drives policy action
        # sampling). Without this, fitting this reducer mid-training would
        # silently perturb the policy's exploration RNG stream. seed=None
        # preserves the old (shared-stream) behavior for backward compat.
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, n_components)
        )
        
    def fit(self, X: np.ndarray, labels: np.ndarray = None) -> 'ContrastiveEncoderReducer':
        if labels is None:
            raise ValueError("ContrastiveEncoderReducer requires labels for fitting.")
            
        dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(labels))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self._generator)
        
        optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        
        self.encoder.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                if len(batch_y) < 2:
                    continue
                optimizer.zero_grad()
                
                # Forward pass
                z = self.encoder(batch_x)
                # Normalize for cosine similarity
                z = nn.functional.normalize(z, p=2, dim=1)
                
                # Compute similarity matrix
                sim_matrix = torch.matmul(z, z.T) / self.temperature
                
                # Loss computation (InfoNCE-style) - Vectorized
                B = len(batch_y)
                labels_eq = (batch_y.unsqueeze(0) == batch_y.unsqueeze(1))
                pos_mask = labels_eq.clone()
                pos_mask.fill_diagonal_(False)
                
                if pos_mask.any():
                    neg_mask = ~labels_eq
                    exp_sims = torch.exp(sim_matrix)
                    denom = torch.sum(exp_sims * neg_mask.float(), dim=1)
                    
                    loss_matrix = -sim_matrix + torch.log(exp_sims + denom.unsqueeze(1) + 1e-8)
                    loss = torch.sum(loss_matrix * pos_mask.float())
                    
                    loss = loss / B
                    loss.backward()
                    optimizer.step()
                    
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(torch.FloatTensor(X))
            z = nn.functional.normalize(z, p=2, dim=1)
        return z.numpy()


class MixerReducer(BaseDimensionalityReducer):
    """
    Non-linear mixer that takes concatenated marginal occupancies and
    outputs a scalar win-alignment logit. Replaces both reducer and scorer.
    
    Architecture:
        MLP: R^{Nd} -> 128 -> 64 -> 1
    
    Training:
        BCE loss on win/loss labels + variance regularization to prevent
        within-class collapse.
        Labels: y=1 for win_buffer signatures, y=0 for loss_buffer signatures.
    
    Output interpretation:
        ψ(Z) = log P(win|Z) / P(loss|Z)  (log-odds, from BCE optimum)
        ψ > 0: win-like behaviour
        ψ < 0: loss-like behaviour
    
    Interaction terms arise because the hidden layers mix all agents' 
    occupancy blocks through shared weights. The counterfactual difference
    ψ(Z) - ψ(Z^{-i}) is non-zero even for agents j≠i because the 
    non-linear hidden layers entangle all blocks.
    """
    def __init__(self, input_dim: int, lr: float = 0.001,
                 epochs: int = 20, batch_size: int = 64,
                 var_reg: float = 0.01, seed: int = None):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.var_reg = var_reg
        # Dedicated generator for DataLoader shuffling, isolated from
        # torch's global default generator (which drives policy action
        # sampling). Without this, fitting this reducer mid-training would
        # silently perturb the policy's exploration RNG stream. seed=None
        # preserves the old (shared-stream) behavior for backward compat.
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)   # scalar logit output
        )
    
    def fit(self, X: np.ndarray, labels: np.ndarray = None):
        """
        Train the mixer on win/loss labeled signatures.
        
        Args:
            X: (n_samples, Nd) concatenated occupancy signatures
            labels: (n_samples,) binary labels, 1=win, 0=loss
        """
        if labels is None:
            raise ValueError("MixerReducer requires labels for fitting.")
        
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(labels)
        
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self._generator)

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        self.net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                if len(batch_y) < 2:
                    continue
                
                optimizer.zero_grad()
                
                logits = self.net(batch_x).squeeze(-1)  # (batch,)
                
                # BCE loss
                loss_bce = criterion(logits, batch_y)
                
                # Variance regularization: prevent within-class collapse
                win_mask = (batch_y == 1)
                loss_mask = (batch_y == 0)
                loss_var = torch.tensor(0.0)
                if win_mask.sum() > 1:
                    loss_var = loss_var - torch.var(logits[win_mask])
                if loss_mask.sum() > 1:
                    loss_var = loss_var - torch.var(logits[loss_mask])
                
                loss = loss_bce + self.var_reg * loss_var
                
                loss.backward()
                optimizer.step()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Returns scalar logit scores, shape (batch, 1).
        """
        self.net.eval()
        with torch.no_grad():
            scores = self.net(torch.FloatTensor(X))
        return scores.numpy()   # (batch, 1)


class TrajectoryReturnRegressor(BaseDimensionalityReducer):
    """Predict a scalar episode target from a joint signature.

    This is a regression alternative to :class:`MixerReducer`, not a
    replacement for it. Both models consume the same concatenated per-agent
    episode representation and produce a scalar score, which lets the
    existing counterfactual shaper compare factual and agent-replaced joint
    signatures without any special-case scoring code.

    Unlike the BCE mixer, this model fits the supplied numeric labels with
    mean-squared error. Controlled Pipelines 2 and 5 supply their selected
    episode scores as targets. The final layer is linear to avoid a
    target-specific output bound.
    """

    def __init__(self, input_dim: int, lr: float = 0.001,
                 epochs: int = 20, batch_size: int = 64,
                 seed: int = None, target_scale: float = 1.0):
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("Target scale must be positive and finite")
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.target_scale = float(target_scale)
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

        # Keep capacity identical to MixerReducer so experiments isolate the
        # supervised objective (return regression versus BCE classification).
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def fit(self, X: np.ndarray, labels: np.ndarray = None):
        """Fit the scalar scorer to the supplied episode targets."""
        if labels is None:
            raise ValueError("TrajectoryReturnRegressor requires numeric return targets.")

        X_t = torch.as_tensor(X, dtype=torch.float32)
        returns_t = torch.as_tensor(labels, dtype=torch.float32).reshape(-1) * self.target_scale
        if len(X_t) != len(returns_t):
            raise ValueError("X and return targets must contain the same number of samples.")
        if not torch.isfinite(returns_t).all():
            raise ValueError("Return targets must be finite.")

        dataset = TensorDataset(X_t, returns_t)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self._generator,
        )
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for batch_x, batch_returns in dataloader:
                # BatchNorm cannot estimate variance from a singleton batch.
                if len(batch_returns) < 2:
                    continue
                optimizer.zero_grad()
                predictions = self.net(batch_x).squeeze(-1)
                loss = criterion(predictions, batch_returns)
                loss.backward()
                optimizer.step()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return predicted episode values with shape ``(batch, 1)``."""
        self.net.eval()
        with torch.no_grad():
            values = self.net(torch.as_tensor(X, dtype=torch.float32))
        return values.numpy() / self.target_scale


class MonotonicMixerReducer(BaseDimensionalityReducer):
    """
    QMIX-style monotonic mixer over occupancy blocks.
    
    Architecture:
        1. Per-agent projector: g_i: R^d -> R  (one per agent)
        2. Hypernetwork: generates non-negative mixing weights w(Z) >= 0
           from the full joint context Z
        3. Output: ψ(Z) = Σ_i w_i(Z) · g_i(φ_i) + b(Z)
    
    Monotonicity guarantee:
        Because w_i >= 0 (enforced by abs()), ∂ψ/∂g_i >= 0.
        If g_i(φ_i) > g_i(φ̄_i), then R_int(i) >= 0.
        Sign of credit is determined by the agent's own performance.
        Magnitude is modulated by cross-agent interactions via w(Z).
    """
    def __init__(self, per_agent_dim: int, n_agents: int,
                 embed_dim: int = 32, lr: float = 0.001,
                 epochs: int = 20, batch_size: int = 64,
                 var_reg: float = 0.01, seed: int = None):
        self.per_agent_dim = per_agent_dim
        self.n_agents = n_agents
        self.input_dim = per_agent_dim * n_agents
        self.embed_dim = embed_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.var_reg = var_reg
        # Dedicated generator for DataLoader shuffling, isolated from
        # torch's global default generator (which drives policy action
        # sampling). Without this, fitting this reducer mid-training would
        # silently perturb the policy's exploration RNG stream. seed=None
        # preserves the old (shared-stream) behavior for backward compat.
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

        # Per-agent projectors: φ_i -> scalar
        self.agent_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(per_agent_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1)
            ) for _ in range(n_agents)
        ])
        
        # Hypernetwork: Z -> non-negative weights
        self.hyper_w = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents)
        )
        
        # Hypernetwork: Z -> bias
        self.hyper_b = nn.Sequential(
            nn.Linear(self.input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
    
    def _forward(self, X_t: torch.Tensor) -> torch.Tensor:
        """X_t: (batch, N*d) -> (batch, 1)"""
        blocks = X_t.reshape(-1, self.n_agents, self.per_agent_dim)
        
        # Per-agent scores
        q_i = [self.agent_projectors[i](blocks[:, i]) 
               for i in range(self.n_agents)]
        q_stack = torch.cat(q_i, dim=1)  # (batch, N)
        
        # Context-dependent mixing weights (non-negative)
        w = torch.abs(self.hyper_w(X_t))  # (batch, N), enforced >= 0
        b = self.hyper_b(X_t)             # (batch, 1)
        
        # Mixed score
        score = (w * q_stack).sum(dim=1, keepdim=True) + b  # (batch, 1)
        return score
    
    def fit(self, X: np.ndarray, labels: np.ndarray = None):
        """Train with BCE + variance regularization."""
        if labels is None:
            raise ValueError("MonotonicMixerReducer requires labels.")
        
        X_t = torch.FloatTensor(X)
        y_t = torch.FloatTensor(labels)
        
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=self._generator)

        all_params = (list(self.agent_projectors.parameters()) +
                      list(self.hyper_w.parameters()) + 
                      list(self.hyper_b.parameters()))
        optimizer = optim.Adam(all_params, lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        for epoch in range(self.epochs):
            for batch_x, batch_y in dataloader:
                if len(batch_y) < 2:
                    continue
                
                optimizer.zero_grad()
                logits = self._forward(batch_x).squeeze(-1)
                
                loss_bce = criterion(logits, batch_y)
                
                win_mask = (batch_y == 1)
                loss_mask = (batch_y == 0)
                loss_var = torch.tensor(0.0)
                if win_mask.sum() > 1:
                    loss_var = loss_var - torch.var(logits[win_mask])
                if loss_mask.sum() > 1:
                    loss_var = loss_var - torch.var(logits[loss_mask])
                
                loss = loss_bce + self.var_reg * loss_var
                
                loss.backward()
                optimizer.step()
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            scores = self._forward(torch.FloatTensor(X))
        return scores.numpy()  # (batch, 1)


class IdentityReducer(BaseDimensionalityReducer):
    def fit(self, X: np.ndarray, labels: np.ndarray = None) -> 'IdentityReducer':
        return self
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        return X
