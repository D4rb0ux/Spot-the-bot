#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import ipdb
from scipy.spatial.distance import pdist, squareform
import pandas as pd


def greedy(X, S, k):
    # remember that k = B * k here...

    M = [np.random.permutation(S)[0]] # M = {m_1}, a random point in S
    # print M

    A = np.setdiff1d(S, M) # A = S \ M
    dists = np.zeros(len(A))

    for i in range(len(A)):
        dists[i] = np.linalg.norm(X[A[i]] - X[M[0]]) # euclidean distance

    # print dists

    for i in range(1, k):
        # choose medoid m_i as the farthest from previous medoids

        midx = np.argmax(dists)
        mi = A[midx]
        
        M.append(mi)
                
        # update the distances, so they reflect the dist to the closest medoid:
        for j in range(len(A)):
            dists[j] = min(dists[j], np.linalg.norm(X[A[j]] - X[mi]))
        
        # remove mi entries from A and dists:
        A = np.delete(A, midx)
        dists = np.delete(dists, midx)

    return np.array(M)


def computeBadMedoids(X, assigns, Dis, Mcurr, minDeviation):
    N, d = X.shape
    k = len(Mcurr)
    Mbad = []
    counts = [len(np.where(assigns == i)[0]) for i in Mcurr]
    cte = int(np.ceil((N / k) * minDeviation))

    # get the medoid with least points:
    Mbad.append(Mcurr[np.argsort(counts)[0]])

    for i in range(len(counts)):
        if counts[i] < cte and Mcurr[i] not in Mbad:
            Mbad.append(Mcurr[i])

    return Mbad


def findDimensions(X, k, l, L, Mcurr):
    N, d = X.shape
    Dis = [] # dimensions picked for the clusters

    Zis = [] # Z for the remaining dimensions
    Rem = [] # remaining dimensions
    Mselidx = [] # id of the medoid indexing the dimensions in Zis and Rem

    for i in range(len(Mcurr)):
        mi = Mcurr[i]
        # Xij is the average distance from the points in L_i to m_i
        # Xij here is an array, containing the avg dists in each dimension
        Xij = np.abs(X[L[i]] - X[mi]).sum(axis = 0) / len(L[i])
        Yi = Xij.sum() / d # average distance over all dimensions
        Di = [] # relevant dimensions for m_i
        si = np.sqrt(((Xij - Yi)**2).sum() / (d-1)) # standard deviations
        Zij = (Xij - Yi) / si # z-scores of distances

        # pick the smallest two:
        o = np.argsort(Zij)
        Di.append(o[0])
        Di.append(o[1])
        Dis.append(Di)

        for j in range(2,d):
            Zis.append(Zij[o[j]])
            Rem.append(o[j])
            Mselidx.append(i)

    if l != 2:
        # we need to pick the remaining dimensions

        o = np.argsort(Zis)
        
        nremaining = k * l - k * 2
        # print("still need to pick %d dimensions." % nremaining)

        # we pick the remaining dimensions using a greedy strategy:
        j = 0
        while nremaining > 0:
            midx = Mselidx[o[j]]
            Dis[midx].append(Rem[o[j]])
            j += 1
            nremaining -= 1

    # print("selected:")
    # print(Dis)

    return Dis


def manhattanSegmentalDist(x, y, Ds):
    """ Compute the Manhattan Segmental Distance between x and y considering
        the dimensions on Ds."""
    dist = 0
    for d in Ds:
        dist += np.abs(x[d] - y[d])
    return dist / len(Ds)


def evaluateClusters(X, assigns, Dis, Mcurr):

    upperSum = 0.0

    for i in range(len(Mcurr)):     
        C = X[np.where(assigns == Mcurr[i])[0]] # points in cluster M_i
        Cm = C.sum(axis = 0) / C.shape[0] # cluster centroid
        Ysum = 0.0

        for d in Dis[i]:
            # avg dist to centroid along dim d:
            Ysum += np.sum(np.abs(C[:,d] - Cm[d])) / C.shape[0]
        wi = Ysum / len(Dis[i])

        upperSum += C.shape[0] * wi

    return upperSum / X.shape[0]


def assignPoints(X, Mcurr, Dis):

    assigns = np.ones(X.shape[0]) * -1

    for i in range(X.shape[0]):
        minDist = np.inf
        best = -1
        for j in range(len(Mcurr)):
            dist = manhattanSegmentalDist(X[i], X[Mcurr[j]], Dis[j])
            if dist < minDist:
                minDist = dist
                best = Mcurr[j]

        assigns[i] = best

    return assigns


def plotDataset(X, D):
    plt.clf()
    plt.plot(X[:,D[0]], X[:,D[1]], 'bo')
    plt.xlabel('X' + str(D[0]))
    plt.ylabel('X' + str(D[1]))
    plt.show()
    plt.draw()

def plotClustering(X, M, A, D):
    """ Plot a proclus clustering result.
        X: the data matrix
        M: medoid indices
        A: cluster assignments
        D: dimensions to plot
    """
    plt.clf()
    plt.xlabel('X' + str(D[0]))
    plt.ylabel('X' + str(D[1]))
    d1, d2 = D
    colors = np.empty(X.shape[0], dtype = 'object')
    colors[np.where(A == -1)[0]] = "0.7" # gray for outliers
    picks = ['b', 'g', 'r', 'c', 'm', 'k', 'y', 'lightblue', 'goldenrod','darkmagenta','coral','deepskyblue','lightpink','tan']

    print('cluster counts:')
    print(np.unique(A))
    print([len(np.where(A == i)[0]) for i in np.unique(A)])

    i = 0
    for c in np.setdiff1d(np.unique(A), [-1]):
        if i >= len(picks):
            raise Exception("used more colors than i have...")
        colors[np.where(A == c)[0]] = picks[i]
        i += 1
    
    plt.scatter(X[:,d1], X[:,d2], c = colors.tolist(), marker = 'o', s = 10)
    # plot medoids as orange diamonds:
    plt.plot(X[M,d1], X[M,d2], marker = 'D', mfc = '#FFFF4D', ms = 7, ls = '')
    plt.show()
    plt.draw()


def proclus(X, k = 2, l = 3, minDeviation = 0.1, A = 30, B = 15, niters = 30, seed = 1234):
    """ Run PROCLUS on a database to obtain a set of clusters and 
        dimensions associated with each one.
        Parameters:
        ----------
        - X:            the data set
        - k:            the desired number of clusters
        - l:            average number of dimensions per cluster
        - minDeviation: for selection of bad medoids
        - A:            constant for initial set of medoids
        - B:            a smaller constant than A for the final set of medoids
        - niters:       maximum number of iterations for the second phase
        - seed:         seed for the RNG
    """
    np.random.seed(seed)

    N, d = X.shape

    A = 2 * k
    B = k

    if B > A:
        raise Exception("B has to be smaller than A.")

    if l < 2:
        raise Exception("l must be >=2.")

    ###############################
    # 1.) Initialization phase
    ###############################

    # first find a superset of the set of k medoids by random sampling
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    S = idxs[0:(A*k)]
    M = greedy(X, S, B * k)
    
    ###############################
    # 2.) Iterative phase
    ###############################

    BestObjective = np.inf

    # choose a random set of k medoids from M:
    Mcurr = np.random.permutation(M)[0:k] # M current
    Mbest = None # Best set of medoids found
    
    try:
        D = squareform(pdist(X)) # precompute the euclidean distance matrix
    except:
        return ("oop", "oop", "oop")

    it = 0 # iteration counter
    L = [] # locality sets of the medoids, i.e., points within delta_i of m_i.
    Dis = [] # important dimensions for each cluster
    assigns = [] # cluster membership assignments

    while True:
        it += 1
        L = []

        for i in range(len(Mcurr)):
            mi = Mcurr[i]
            # compute delta_i, the distance to the nearest medoid of m_i:
            di = D[mi,np.setdiff1d(Mcurr, mi)].min()
            # compute L_i, points in sphere centered at m_i with radius d_i
            L.append(np.where(D[mi] <= di)[0])

        # find dimensions:
        Dis = findDimensions(X, k, l, L, Mcurr)
        
        # form the clusters:
        assigns = assignPoints(X, Mcurr, Dis)
        
        # evaluate the clusters:
        ObjectiveFunction = evaluateClusters(X, assigns, Dis, Mcurr)

        badM = [] # bad medoids

        Mold = Mcurr.copy()

        if ObjectiveFunction < BestObjective:
            BestObjective = ObjectiveFunction
            Mbest = Mcurr.copy()
            # compute the bad medoids in Mbest:
            badM = computeBadMedoids(X, assigns, Dis, Mcurr, minDeviation)

        if len(badM) > 0:
            # replace the bad medoids with random points from M:
            Mavail = np.setdiff1d(M, Mbest)
            newSel = np.random.choice(Mavail, size = len(badM), replace = False)
            Mcurr = np.setdiff1d(Mbest, badM)
            Mcurr = np.union1d(Mcurr, newSel)

       # print("finished iter: %d" % it)

        if np.allclose(Mold, Mcurr) or it >= niters:
            break

    #print("finished iterative phase...")

    ###############################
    # 3.) Refinement phase
    ###############################

    # compute a new L based on assignments:
    L = []
    for i in range(len(Mcurr)):
        mi = Mcurr[i]
        L.append(np.where(assigns == mi)[0])

    Dis = findDimensions(X, k, l, L, Mcurr)
    assigns = assignPoints(X, Mcurr, Dis)

    # handle outliers:

    # smallest Manhattan segmental distance of m_i to all (k-1)
    # other medoids with respect to D_i:
    deltais = np.zeros(k)
    for i in range(k):
        minDist = np.inf
        for j in range(k):
            if j != i:
                dist = manhattanSegmentalDist(X[Mcurr[i]], X[Mcurr[j]], Dis[i])
                if dist < minDist:
                    minDist = dist
        deltais[i] = minDist

    # mark as outliers the points that are not within delta_i of any m_i:
    for i in range(len(assigns)):
        clustered = False
        for j in range(k):
            d = manhattanSegmentalDist(X[Mcurr[j]], X[i], Dis[j])
            if d <= deltais[j]:
                clustered = True
                break
        if not clustered:
            #print "marked an outlier"
            assigns[i] = -1

    return (Mcurr, Dis, assigns)


def I_index(data, centroids, c_membership): # maximize
    data = pd.DataFrame(data)
    medoids = pd.DataFrame()
    for c in centroids:
        medoids = medoids.append(data.iloc[c,])

    C = data.mean()
    sum = 0
    for index, row in data.iterrows():
        d = np.linalg.norm(C - row)
        sum += d
    
    c_membership = c_membership.astype(int)
    data['cluster'] = c_membership
    n_c = list(set(c_membership))
    n_c.remove(-1)
    sum_ = 0
    for i in n_c:
        sum_c = 0
        for index, row in data[data['cluster'] == i].loc[:, data.columns != 'cluster'].iterrows():
            sum_c += np.linalg.norm(medoids.loc[[i]] - row)
        sum_ += sum_c

    max_dc = 1.0
    for i in n_c:
        for j in n_c:
            dc = np.linalg.norm(medoids.loc[[i]] - medoids.loc[[j]])
            if dc > max_dc:
                max_dc = dc
                
    return ((1.0 / float(len(n_c))) * (sum / sum_) * max_dc)**2


from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score


X = np.load('/home/fibykov/veronica/datasetRU_k6n3_even_512.npy', allow_pickle=True)
print("k6n3_even_512")
print("\nall vectors in dataset: ", X.shape[0])
X = np.unique(X, axis=0)
print("unique vectors in dataset: ", X.shape[0])
print()
results = pd.DataFrame(columns = ['k','l','S', 'DB', 'I'])

import time
t0 = time.time()
i = 0 

for k in range(2, 51, 2):
    for l in range(2, 18, 3):
    rseed = np.random.randint(low = 0, high = 1239831)
    M, D, A = proclus(X, k = k, l = l, niters=50, seed = rseed)
    
    if M == "oop":
        print("\nbroke on k = ", k, "\n")
        break
        
    S, DB, I = silhouette_score(X, A), davies_bouldin_score(X, A), I_index(X, M, A)
        #print("k =", k, ", l =", l)
        #print("[max] Silhouette score: ", silhouette_score(X, A))
        #print("[min] Davies Bouldin score: ", davies_bouldin_score(X, A))
        #print("[max] I Index: ", I_index(X, M, A))
    results.loc[i] = [int(k), int(l), S, DB, I]
    i += 1

t1 = time.time()
print("seconds:", (t1 - t0))
print("minutes:", (t1 - t0) / 60)
print("hours:", (t1 - t0) / 3600)

results['k'] = results['k'].astype('int')
results['l'] = results['l'].astype('int')
print("\nBest according to Silhouette:")
print(results[results.S == results.S.max()])
print("\nBest according to Davies Bouldin:")
print(results[results.DB == results.DB.min()])
print("\nBest according to I Index:")
print(results[results.I == results.I.max()])
print()

results.to_csv('results.csv', index=False)
print(results)

