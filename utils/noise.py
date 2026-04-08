import numpy as np

def make_noise_personality(num_users, num_feats, dist, seed=42):
    """
    Generate synthetic personality features with different distributions
    
    Args:
        num_users: Number of users
        num_feats: Number of personality features
        dist: Distribution type ('uniform', 'normal', 'laplace', 'bernoulli', 'exponential')
        seed: Random seed
    
    Returns:
        Numpy array of shape (num_users, num_feats)
    """
    rng = np.random.RandomState(seed)
    
    if dist == "uniform":
        return rng.uniform(1, 10, size=(num_users, num_feats))
    
    elif dist == "normal":
        return rng.normal(loc=5.0, scale=2.0, size=(num_users, num_feats))
    
    elif dist == "laplace":
        return rng.laplace(loc=5.0, scale=1.5, size=(num_users, num_feats))
    
    elif dist == "bernoulli":
        return rng.binomial(n=1, p=0.6, size=(num_users, num_feats)).astype(float)
    
    elif dist == "exponential":
        return rng.exponential(scale=4.0, size=(num_users, num_feats))
    
    else:
        raise ValueError(f"Unknown distribution: {dist}")