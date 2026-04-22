import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel
import faiss


class GreedyCoreset:
    def __init__(self, corpus,labels, metric,n_neighbors):
        self.corpus = np.array(corpus, dtype=np.float32)  #cause fais needs flaot32
        self.labels = labels
        self.n = len(corpus)
        self.metric = metric
        self.n_neighbors = n_neighbors
        # Build FAISS index
        d = self.corpus.shape[1]  # 768
        if metric == 'cosine':
            # normalize first, then use inner product (= cosine on normalized vecs)
            faiss.normalize_L2(self.corpus)
            self.index = faiss.IndexFlatIP(d)
        else:  # euclidean
            self.index = faiss.IndexFlatL2(d)
        
        # Move to GPU if available
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        
        self.index.add(self.corpus)

    def _dist(self, a, b):
                if self.metric == "euclidean":
                    return np.linalg.norm(a - b)
                elif self.metric == "cosine":
                    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    def _compute_all_dists(self, point):
        if self.metric == 'euclidean':
            return np.linalg.norm(self.corpus - point, axis=1)
        elif self.metric == 'cosine':
            point_norm = point / (np.linalg.norm(point) + 1e-10)
            return 1 - self.corpus @ point_norm  
    def select(self, corset_size, sample_size, per_label=False): #what you cocall
        
            print(sample_size,self.n_neighbors,self.metric)
            return self._select(corset_size, sample_size)
    
    def _select(self, corset_size, sample_size):  #the core corset selection algo 
        C = []
        set_c=set()
        # array to keep for each datapoint in corpus the min distance we have from a point to it
        #initialse with sth huge
        min_dists = np.full(self.n, np.inf)
        """
        min_dist is storing for each point the closest representative to it from points from C
        """
        
        # This section of code for the initialisation of the first point to start working with 
        if corset_size > 0:
            # Pick point farthest from mean (diverse start)= fixed methode 
            mean = np.mean(self.corpus, axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-10)
            first_idx = int(np.argmax(self._compute_all_dists(mean)))
            C.append(first_idx)  
            set_c.add(first_idx)
            # Update min distances 
            min_dists = self._compute_all_dists(self.corpus[first_idx])

        #-----------------------------------------------------------------------------
        while len(C) < corset_size:
            # Sample candidates (not in C)
            """ candidates = list(set(range(self.n)) - set(C))
            if len(candidates) > sample_size:
                candidates = random.sample(candidates, sample_size)"""
            candidates=get_samples(self.corpus, set_c, size_samples=sample_size)
            best_t = None
            best_utility = -np.inf
            
            for t in candidates:
                # Fast utility computation using nearest neighbors
                # Get distances from t to all points (approximate)
            
                query = self.corpus[t].reshape(1, -1)
                dist_to_t, indexes = self.index.search(query, self.n_neighbors)
                dist_to_t = dist_to_t[0]
                indexes = indexes[0]
                if self.metric == 'cosine':
                        dist_to_t = 1 - dist_to_t
                if self.metric == 'euclidean':
                        dist_to_t = np.sqrt(dist_to_t) 
                # Compute reduction
                reduction = 0
                l_neighbors=len(dist_to_t)
                """
                amoung the k closest neighbors we picked we are gonna see 
                which of them is the cloest to the rest of datapoitns
                from the main corpus 
                """
                for j, i in enumerate(indexes): # i is dataset index  and j positiion in neigh list 
                    if i in set_c:  #if this sample is already selected
                        continue
                    if dist_to_t[j] < min_dists[i]:
                        #this bewlo is teh utility
                        # E(c) -E[c+t]
                        reduction += min_dists[i] - dist_to_t[j]
                #reduction is the "imporvment " it will bring to the corset
                #
                if reduction > best_utility:
                    best_utility = reduction
                    best_t = t
            
            if best_t is not None:
                   C.append(best_t)
                   set_c.add(best_t)
                   dists_to_new = self._compute_all_dists(self.corpus[best_t])
                   min_dists = np.minimum(min_dists, dists_to_new)


            if len(C)%100==0:
                    print(f"Selected {len(C)}/{corset_size} tuples")
        
        return C
    
    
    
   

def get_samples(corpus, C_set, size_samples):
    candidates = list(set(range(len(corpus))) - C_set)
    if len(candidates) <= size_samples:
        return candidates
    return random.sample(candidates, size_samples)

def save_corset_indexes(coreset_indices,corset_size,sample_size,metric,knn):
    np.save(f"coreset{sample_size}_{corset_size}_{metric}_{knn}.npy", coreset_indices)
def restore_corset(file_name,Y,X):
    indexes=np.load(file_name)     
    indexes = indexes.astype(int).tolist()
    X_coreset = X[indexes]
    Y_coreset = Y[indexes] 
    return X_coreset, Y_coreset
