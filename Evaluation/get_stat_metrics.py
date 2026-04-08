

from goodCore.Experiments.Code.utils import load_embeddings
"""
Nothing is being saved to files yet
"""
def run_and_save_stat_tests(features_corpus, features_coreset, Y, Y_coreset):
    r1 = get_labels_freq(Y, Y_coreset)
    r2 = get_kl_divergence(Y_coreset, Y)          # drops the file save for now
    r3 = hausdorff_coverage(features_corpus, features_coreset)
    r4 = mmd_rbf(features_corpus, features_coreset)
    r5 = diversity_ratio(features_corpus, features_coreset)
    r6, _ = wasserstein_pca(features_corpus, features_coreset)

    return {
        'label_freq'      : r1,
        'kl'              : r2,
        'hausdorff'       : r3,
        'mmd'             : r4,
        'diversity'       : r5,
        'wasserstein_pca' : r6,
    }




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


def wasserstein_pca(corpus_features, coreset_features, n_components=10):
    """
    takes the features embeddings  
    """
    pca         = PCA(n_components=n_components)
    full_proj   = pca.fit_transform(corpus_features)
    coreset_proj= pca.transform(coreset_features)

    results = []
    for i in range(n_components):
        w = wasserstein_distance(full_proj[:, i], coreset_proj[:, i])
        results.append({
            'pc'             : i + 1,
            'explained_var'  : float(pca.explained_variance_ratio_[i]),
            'wasserstein'    : float(w)
        })
    return results, pca


def plot_embedding_coverage(corpus_features, coreset_features, hausdorff_res, wass_res):
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # — PCA scatter ————————————————————————————————
    pca  = PCA(n_components=2)
    fp   = pca.fit_transform(corpus_features)
    cp   = pca.transform(coreset_features)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(fp[:, 0], fp[:, 1], alpha=0.1, s=4,
                c='steelblue', label=f'Full (n={len(corpus_features)})')
    ax1.scatter(cp[:, 0], cp[:, 1], alpha=0.8, s=18,
                c='crimson', edgecolors='darkred', linewidths=0.3,
                label=f'Coreset (k={len(coreset_features)})')
    ax1.set_title('PCA — embedding coverage')
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
    ax1.legend(fontsize=8)

    # — Coverage distance histogram ————————————————
    ax2 = fig.add_subplot(gs[0, 1])
    dists = hausdorff_res['dists']
    ax2.hist(dists, bins=60, color='steelblue', alpha=0.75, edgecolor='white')
    ax2.axvline(hausdorff_res['mean_coverage'], color='orange',
                linestyle='--', lw=1.5, label=f"mean={hausdorff_res['mean_coverage']:.3f}")
    ax2.axvline(hausdorff_res['hausdorff'], color='crimson',
                linestyle='--', lw=1.5, label=f"hausdorff={hausdorff_res['hausdorff']:.3f}")
    ax2.axvline(hausdorff_res['p90_coverage'], color='purple',
                linestyle='--', lw=1.5, label=f"p90={hausdorff_res['p90_coverage']:.3f}")
    ax2.set_title('Coverage distance distribution')
    ax2.set_xlabel('dist to nearest coreset point')
    ax2.legend(fontsize=8)

    # — Wasserstein per PC ————————————————————————
    ax3 = fig.add_subplot(gs[0, 2])
    pcs   = [r['pc']          for r in wass_res]
    wvals = [r['wasserstein'] for r in wass_res]
    evars = [r['explained_var'] * 100 for r in wass_res]
    colors = ['crimson' if w > np.mean(wvals) + np.std(wvals) else 'steelblue'
              for w in wvals]
    bars = ax3.bar([f'PC{p}' for p in pcs], wvals, color=colors, alpha=0.8)
    ax3b = ax3.twinx()
    ax3b.plot([f'PC{p}' for p in pcs], evars,
              color='orange', marker='o', ms=4, lw=1.5, label='explained var %')
    ax3b.set_ylabel('explained variance %', fontsize=8)
    ax3.set_title('Wasserstein per PC')
    ax3.set_ylabel('Wasserstein distance')
    ax3.tick_params(axis='x', labelsize=8)
    ax3b.legend(fontsize=8)
    plt.suptitle(f'Embedding coverage — K={len(coreset_features)} / N={len(corpus_features)}',
                 fontsize=12)
    plt.savefig('embedding_coverage.png', dpi=150, bbox_inches='tight')
    plt.show()

