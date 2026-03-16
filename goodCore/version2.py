import numpy as np
import random
from sklearn.neighbors import NearestNeighbors



"""
Fixed : the utility computing using min_dist instead of recomputing distances for every new point t

"""
class GreedyCoreset:
    def __init__(self, corpus, labels=None):
        self.corpus = np.array(corpus)
        self.labels = labels
        self.n = len(corpus)
        
        # you use nn later to get the 100 nearest neighbors for each dp
        self.nn = NearestNeighbors(n_neighbors=min(100, self.n), metric='euclidean')
        self.nn.fit(self.corpus)
    
    def select(self, K=50, sample_size=100, per_label=True): #what you cocall
        if per_label and self.labels is not None:
            return self._select_per_label(K, sample_size)
        else:
            return self._select(K, sample_size)
    
    def _select(self, K, sample_size):  #the core corset selection algo 
        C = []
        
        # array to keep for each datapoint in corpus the min distance we have from a point to it
        #initialse with sth huge
        min_dists = np.full(self.n, np.inf)
        """
        min_dist is storing for each point the closest representative to it from points from C
        """
        
        # First point: pick farthest from center or random
        if K > 0:
            # Pick point farthest from mean (diverse start)
            mean = np.mean(self.corpus, axis=0)
            first_idx = np.argmax([np.linalg.norm(x - mean) for x in self.corpus])
            C.append(first_idx)
            
            # Update min distances
            for i in range(self.n):
                min_dists[i] = np.linalg.norm(self.corpus[i] - self.corpus[first_idx])
        
        while len(C) < K:
            # Sample candidates (not in C)
            """ candidates = list(set(range(self.n)) - set(C))
            if len(candidates) > sample_size:
                candidates = random.sample(candidates, sample_size)"""
            candidates=get_samples(self.corpus, C, size_samples=100)
            best_t = None
            best_utility = -np.inf
            
            for t in candidates:
                # Fast utility computation using nearest neighbors
                # Get distances from t to all points (approximate)
                dist_to_t, indexes = self.nn.kneighbors(
                    self.corpus[t].reshape(1, -1), 
                  #  n_neighbors=self.n,  #why specify n == everyone 
                    return_distance=True
                )
                dist_to_t = dist_to_t[0]
                indexes = indexes[0]

                # Compute reduction
                reduction = 0
                l_neighbors=len(dist_to_t)
                """
                amoung the k closest neighbors we picked we are gonna see 
                which of them is the cloest to the rest of datapoitns
                from the main corpus 
                """
                for j, i in enumerate(indexes): # i is dataset index  and j positiion in neigh list 
                    if i in C:
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
                
                # Update min distances
                for i in range(self.n):
                    if i in C:
                        continue
                    """
                    here you added new point t to the corset , and remember that mindist 
                    is computing dis based on points in the corset 
                    since you added new point to corset recompute to see 
                    """
                    #PP
                    dist = np.linalg.norm(self.corpus[i] - self.corpus[best_t])
                    if dist < min_dists[i]:
                        min_dists[i] = dist
            
            print(f"Selected {len(C)}/{K} tuples")
        
        return C
    
    def _select_per_label(self, K, sample_size):
        unique_labels = np.unique(self.labels)
        C = []
        
        for label in unique_labels:
            label_indices = np.where(self.labels == label)[0]
            label_ratio = len(label_indices) / self.n
            label_K = max(1, int(K * label_ratio))
            
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

