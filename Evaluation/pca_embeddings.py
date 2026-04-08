

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
