
import numpy as np


"""
Label frequency   get_labels_freq
hausdorff label freq :hausdorff_coverage 
MMD : mmd_rbf
diversity ratio : diversity_ratio
kl divergence : get_kl_divergence


! DO NOt save to file , just return , kl_div saves to file and prints
"""
def get_labels_freq(Y, Y_coreset):

        return f"""
        Original dataset label frequencies: {np.sum(Y, axis=0)/len(Y)}
        --------------------------------------------------------------
        Corset dataset label frequencies: {np.sum(Y_coreset, axis=0)/len(Y_coreset)}
        """


def hausdorff_coverage(corpus_features, coreset_features):
    """
        Takes the embeddings of corset and corpus and computes the Hausdorff distance and coverage metrics.
    """
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(coreset_features)
    dists, _ = nn.kneighbors(corpus_features)
    dists    = dists.flatten()
    return {
        'hausdorff'     : float(np.max(dists)),   # THE number — worst gap
        'mean_coverage' : float(np.mean(dists)),  # average gap
        'std_coverage'  : float(np.std(dists)),   # uniformity of coverage
        'p90_coverage'  : float(np.percentile(dists, 90)),  # robust worst-case
        'dists'         : dists
    }

def mmd_rbf(corpus_features, coreset_features, max_samples=2000):
        
    """ Computes the MMD between corpus and coreset using an RBF kernel with median heuristic bandwidth.
        Returns MMD², MMD, and the gamma used.
    """     
    rng = np.random.default_rng(42)

    # subsample full corpus if large
    idx = rng.choice(len(corpus_features), size=min(max_samples, len(corpus_features)), replace=False)
    #is doing tf what above?
    X   = corpus_features[idx].astype(np.float32)
    Y   = coreset_features.astype(np.float32)

    # median heuristic for bandwidth: γ = 1 / (2 · median_dist²)
    sample   = np.vstack([X[:500], Y[:500]])
    pairwise = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
    median_sq = np.median(pairwise[pairwise > 0])
    gamma     = 1.0 / (2.0 * median_sq)

    K_XX = rbf_kernel(X, X, gamma=gamma)
    K_YY = rbf_kernel(Y, Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    mmd2 = K_XX.mean() - 2 * K_XY.mean() + K_YY.mean()
    return {
        'mmd2'  : float(mmd2),              # MMD² — main number
        'mmd'   : float(np.sqrt(max(mmd2, 0))),  # MMD  — easier to read
        'gamma' : float(gamma)
    }
def diversity_ratio(corpus_features, coreset_features, max_samples=2000):
    """ Computes the diversity ratio: mean pairwise distance in coreset / mean pairwise distance in corpus.
                Subsamples the corpus if it's too large for efficiency.
                Returns the mean pairwise distances and the ratio.
    """
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(42)

    c_pdist = pairwise_distances(coreset_features, metric='euclidean')
    np.fill_diagonal(c_pdist, np.nan)
    mean_coreset = float(np.nanmean(c_pdist))

    idx     = rng.choice(len(corpus_features), size=min(max_samples, len(corpus_features)), replace=False)
    f_pdist = pairwise_distances(corpus_features[idx], metric='euclidean')
    np.fill_diagonal(f_pdist, np.nan)
    mean_full = float(np.nanmean(f_pdist))

    return {
        'mean_pairwise_coreset' : mean_coreset,
        'mean_pairwise_full'    : mean_full,
        'diversity_ratio'       : mean_coreset / mean_full
    }



def get_kl_divergence(Y_coreset, Y, output_file="kl_results.txt"):
    """

     Not embeddings , takes the labels  
    """
    CLASS_NAMES = [
        "atelectasis", "cardiomegaly", "effusion",
        "infiltration", "mass", "nodule", "pneumonia",
        "pneumothorax", "consolidation", "edema",
        "emphysema", "fibrosis", "pleural_thickening", "hernia"
    ]
    
    eps = 1e-10
    
    # prevalence per class
    p_full    = np.mean(Y, axis=0) + eps
    p_coreset = np.mean(Y_coreset, axis=0) + eps
    
    # normalize
    p_full    = p_full / p_full.sum()
    p_coreset = p_coreset / p_coreset.sum()
    
    # divergences
    kl = entropy(p_full, p_coreset)
    js = jensenshannon(p_full, p_coreset) ** 2

    # ---- WRITE TO FILE ----
    with open(output_file, "w") as f:
        f.write(f"KL divergence : {kl:.6f}\n")
        f.write(f"JS divergence : {js:.6f}\n\n")
        
        f.write(f"{'class':<22} {'full':>8} {'coreset':>10} {'abs diff':>10}\n")
        f.write("-" * 52 + "\n")
        
        for name, pf, pc in zip(CLASS_NAMES, p_full, p_coreset):
            diff = abs(pf - pc)
            flag = " ◄ check" if diff > 0.01 else ""
            f.write(f"{name:<22} {pf:>8.4f} {pc:>10.4f} {diff:>10.4f}{flag}\n")

    print(f"Results saved to {output_file}")

    # ---- OPTIONAL: still print ----
    print(f"KL divergence : {kl:.4f}")
    print(f"JS divergence : {js:.4f}")

    # ---- PLOT ----
    x = np.arange(len(CLASS_NAMES))
    w = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - w/2, np.mean(Y, axis=0),  w, label="full dataset")
    ax.bar(x + w/2, np.mean(Y_coreset, axis=0), w, label="coreset")
    
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("prevalence")
    ax.set_title(f"label distribution — KL={kl:.4f}  JS={js:.4f}")
    ax.legend()
    
    plt.tight_layout()
    plt.show()