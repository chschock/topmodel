# import click
import sys
import numpy as np
import pandas as pd
import pickle
from fastRecover import nonNegativeRecover
from anchors import findAnchors
from scipy import sparse
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from nonnegfac.nnls import nnlsm_blockpivot

from Q_matrix import generate_Q_matrix
import scipy.io


class Params:
    def __init__(self, filename):
        self.log_prefix = None
        self.checkpoint_prefix = None
        self.seed = int(time.time())

        with open(filename, "rt") as f:
            for l in f:
                if l == "\n" or l[0] == "#":
                    continue
                l = l.strip()
                l = l.split("=")
                if l[0] == "log_prefix":
                    self.log_prefix = l[1]
                elif l[0] == "max_threads":
                    self.max_threads = int(l[1])
                elif l[0] == "eps":
                    self.eps = float(l[1])
                elif l[0] == "checkpoint_prefix":
                    self.checkpoint_prefix = l[1]
                elif l[0] == "new_dim":
                    self.new_dim = int(l[1])
                elif l[0] == "seed":
                    self.seed = int(l[1])
                elif l[0] == "anchor_thresh":
                    self.anchor_thresh = int(l[1])
                elif l[0] == "top_words":
                    self.top_words = int(l[1])


# parse input args
if len(sys.argv) > 5:
    infile = sys.argv[1]
    settings_file = sys.argv[2]
    K = int(sys.argv[3])
    loss = sys.argv[4]
    outfile = sys.argv[5]

else:
    print(
        "usage: ./learn_topics.py word_doc_matrix settings_file K loss output_filename"
    )
    print("for more info see readme.txt")
    sys.exit()

params = Params(settings_file)

corpus = pd.read_csv(infile, header=None, index_col=0, squeeze=True)
cnt_vecr = CountVectorizer(min_df=0.002, max_df=0.2, # max_features=5000, binary=True,
                           token_pattern=r'\b[^(\W|\d)]\w\w\w+\b')
M = cnt_vecr.fit_transform(corpus)
M = M[np.asarray(M.sum(axis=1)).squeeze() > 5, :]  # no pointless docs in training
M = M.T.tocsc().astype(float)

print('{} words / {} documents'.format(*M.shape))

print("identifying candidate anchors")
candidate_anchors = []

# only accept anchors that appear in a significant number of docs
for i in range(M.shape[0]):
    if len(np.nonzero(M[i, :])[1]) > params.anchor_thresh:
        candidate_anchors.append(i)

print(len(candidate_anchors), "candidates")

# forms Q matrix from document-word matrix

# Qorg = generate_Q_matrix(M)
vocab_sz = M.shape[0]
doclengths = np.array(M.sum(axis=0))
H = M.multiply(1 / doclengths)
Q = H.dot(H.T)
Q = Q - sparse.diags(Q.diagonal())
Q = Q.multiply(1 / Q.sum()).toarray()


vocab = cnt_vecr.get_feature_names()

# check that Q sum is 1 or close to it
print("Q sum is", Q.sum())
V = Q.shape[0]
print("done reading documents")

# find anchors- this step uses a random projection
# into low dimensional space
anchors = findAnchors(Q, K, params, candidate_anchors)
print("anchors are:")
for i, a in enumerate(anchors):
    print(i, vocab[a])

# recover topics
#A, topic_likelihoods = nonNegativeRecover(Q, anchors, loss, params)

Q_bar = normalize(Q, axis=1, norm='l1')
Q_anchors = Q[anchors, :]

A_prime, stat = nnlsm_blockpivot(Q_anchors.T, Q.T)
A_prime = A_prime.T
#print(stat)

topic_likelihoods = np.linalg.norm(A_prime, axis=0, ord=1)
A = normalize(A_prime, axis=0, norm='l1')

print("done recovering")

np.savetxt(outfile + ".A", A)
np.savetxt(outfile + ".topic_likelihoods", topic_likelihoods)

B, stat = nnlsm_blockpivot(A, H)
#print(stat)

# display
with open(outfile + ".topwords", "w") as f:
    for k in range(K):
        topwords = np.argsort(A[:, k])[-params.top_words :][::-1]
        print(vocab[anchors[k]], ":", end=" ")
        print(vocab[anchors[k]], ":", end=" ", file=f)
        for w in topwords:
            print(vocab[w], end=" ")
            print(vocab[w], end=" ", file=f)
        print("")
        print("", file=f)

with open(outfile + '.pkl', 'wb') as f:
    pickle.dump((A, B, topic_likelihoods), file=f)

#import ipdb; ipdb.set_trace()
