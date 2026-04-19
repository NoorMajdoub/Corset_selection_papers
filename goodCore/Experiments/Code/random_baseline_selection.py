import numpy as np

def random_subset(X_train, Y_train, fraction=0.05, seed=42, save_path="subset_indices.npy"):
    """
    Randomly samples a fraction of the training set and saves the indices.
    Returns:
        X_subset, Y_subset, indices
    """
    rng = np.random.default_rng(seed)
    
    n_total  = len(X_train)
    n_sample = int(n_total * fraction)
    
    indices = rng.choice(n_total, size=n_sample, replace=False)
    indices = np.sort(indices)  # keep them ordered
    
    X_subset = X_train[indices]
    Y_subset = Y_train[indices]
    
    np.save(save_path, indices)
    
    print(f"Total train samples : {n_total}")
    print(f"Requested fraction  : {fraction:.1%}")
    print(f"Subset size         : {n_sample}")
    print(f"Indices saved to    : {save_path}")
    
    return X_subset, Y_subset, indices


# --- Usage ---
X_sub, Y_sub, idx = sample_train_subset(X_train_s, Y_train_s, fraction=0.05)

# --- Later, to reload ---
idx_loaded = np.load("subset_indices.npy")
X_sub = X_train_s[idx_loaded]
Y_sub = Y_train_s[idx_loaded]
