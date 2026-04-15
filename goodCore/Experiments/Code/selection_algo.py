import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel


class GreedyCoreset:
    def __init__(self, corpus,labels, metric,n_neighbors):
        self.corpus = np.array(corpus)
        self.labels = labels
        self.n = len(corpus)
        
        # you use nn later to get the 100 nearest neighbors for each dp
        #min(100, self.n)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.nn.fit(self.corpus)
    
    def select(self, corset_size=500, sample_size=100, per_label=False): #what you cocall
        if per_label and self.labels is not None:
            return self._select_per_label(corset_size, sample_size)
        else:
            return self._select(corset_size, sample_size)
    
    def _select(self, corset_size, sample_size):
        C = []
        C_set = set()  # for O(1) lookups
        min_dists = np.full(self.n, np.inf)
    
        # Vectorized first point
        mean = np.mean(self.corpus, axis=0)
        first_idx = int(np.argmax(np.linalg.norm(self.corpus - mean, axis=1)))
        C.append(first_idx)
        C_set.add(first_idx)
    
        # Vectorized initial min_dists update
        min_dists = np.linalg.norm(self.corpus - self.corpus[first_idx], axis=1)
    
        while len(C) < corset_size:
            candidates = get_samples(self.corpus, C_set, size_samples=sample_size)
            best_t = None
            best_utility = -np.inf
    
            for t in candidates:
                dist_to_t, indexes = self.nn.kneighbors(
                    self.corpus[t].reshape(1, -1),
                    return_distance=True
                )
                dist_to_t = dist_to_t[0]
                indexes = indexes[0]
    
                # Vectorized reduction — no Python loop over neighbors
                mask = np.array([i not in C_set for i in indexes])
                valid_idx = indexes[mask]
                valid_dists = dist_to_t[mask]
                reduction = np.sum(np.maximum(0, min_dists[valid_idx] - valid_dists))
    
                if reduction > best_utility:
                    best_utility = reduction
                    best_t = t
    
            if best_t is not None:
                C.append(best_t)
                C_set.add(best_t)
    
                # Vectorized min_dists update
                new_dists = np.linalg.norm(self.corpus - self.corpus[best_t], axis=1)
                min_dists = np.minimum(min_dists, new_dists)
                min_dists[best_t] = np.inf  # exclude C points
    
            if len(C) % 100 == 0:
                print(f"Selected {len(C)}/{corset_size} tuples")
    
        return C
        
      
            
            

            
    
    def _select_per_label(self, corset_size, sample_size):
        unique_labels = np.unique(self.labels)  #get all of out labels here they are 
        #update per the frequency of the class
        
        C = []
        
        for label in unique_labels:
            label_indices = np.where(self.labels == label)[0]
            label_ratio = len(label_indices) / self.n
            label_K = max(1, int(corset_size * label_ratio))
            
            if label_K > 0 and len(label_indices) > 0:
                # Create sub-selector for this label
                label_selector = GreedyCoreset(
                    self.corpus[label_indices], 
                    labels=None
                )
                label_C = label_selector._select(label_K, sample_size)
                
                # Map back to original indices
                C.extend(label_indices[label_C].tolist())
        
        return C
    
    def compute_weights(self, C):
        """Compute weights for coreset points"""
        weights = np.zeros(len(C))
        
        for i in range(self.n):
            # Find closest coreset point
            min_dist = np.inf
            closest_idx = -1
            for j, c in enumerate(C):
                dist = np.linalg.norm(self.corpus[i] - self.corpus[c])
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = j
            
            weights[closest_idx] += 1
        
        return weights

def get_samples(corpus, C_set, size_samples):
    candidates = list(set(range(len(corpus))) - C_set)
    if len(candidates) <= size_samples:
        return candidates
    return random.sample(candidates, size_samples)
def get_selector(features_corset, labels_corset,metric='euclidean',n_neighbors=100 ): 
    selector = GreedyCoreset(features_corset, labels_corset,metric,n_neighbors )
    return selector
def select_corset(selector, corset_size=1120, sample_size=100, per_label=False):
    coreset_indices = selector.select(corset_size=1120, sample_size=100)
    return coreset_indices
def save_corset_indexes(coreset_indices,corset_size,sample_size,metric,knn):
    np.save(f"coreset{sample_size}_{corset_size}_{metric}_{knn}.npy", coreset_indices)
def restore_corset(file_name,Y,X):
    indexes=np.load(file_name)     
    indexes = indexes.astype(int).tolist()
    X_coreset = X[indices]
    Y_coreset = Y[indices] 
    return X_coreset, Y_coreset
