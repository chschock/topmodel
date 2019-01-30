import logging
import time
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from sklearn.decomposition.nmf import non_negative_factorization
from sklearn.base import BaseEstimator, TransformerMixin

import random_projection as rp
import gram_schmidt_stable as gs

log = logging.getLogger(__name__)

TOL = 1e-4


def show_topics(topic_term_mat, vocab, P_term, lbd=1, n_top_terms=15):
    """
    ### present topics by relevant words
    - from pyldavis, but footnote 2 there is incorrect
    - https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf
    """
    a = np.asarray(
        lbd * np.log1p(topic_term_mat) +
        (1 - lbd) * np.log1p(np.multiply(topic_term_mat, 1 / P_term.reshape(1, -1)))
    )

    def top_words(t):
        return [vocab[i] for i in np.argsort(t)[:-n_top_terms-1:-1]]

    topic_words = [top_words(t) for t in a]
    topic_list = [' '.join(t) for t in topic_words]
    return topic_list


def get_Q(M):
    """
    https://people.csail.mit.edu/dsontag/papers/AroraEtAl_icml13_supp.pdf Section 4.1
    M: term x document matrix
    Returns coocurrence matrix Q
    """
    doclengths = np.array(M.sum(axis=0))
    H = M.multiply(1 / np.sqrt(np.multiply(doclengths, doclengths - 1)))
    diags = np.array(M.multiply(1 / np.multiply(doclengths, doclengths - 1)).sum(axis=1)).flatten()
    Q = H.dot(H.T) - sparse.diags(diags)
    Q = np.asarray((Q / Q.sum()).todense())
    assert np.isclose(Q[Q < 0], 0).all()
    Q[Q < 0] = 0
    return Q


def find_anchors(Q, K, candidates, dim, seed):
    # Random number generator for generating dimension reduction
    prng_W = np.random.RandomState(seed)

    # row normalize Q
    row_sums = Q.sum(1)
    for i in range(len(Q[:, 0])):
        Q[i, :] = Q[i, :] / float(row_sums[i])

    # Reduced dimension random projection method for recovering anchor words
    Q_red = rp.Random_Projection(Q.T, dim, prng_W)
    Q_red = Q_red.T
    (anchors, anchor_indices) = gs.Projection_Find(Q_red, K, candidates)

    # restore the original Q
    for i in range(len(Q[:, 0])):
        Q[i, :] = Q[i, :] * float(row_sums[i])

    return anchor_indices


def recover_term_topic_matrix(Q, anchors, tol=TOL, betaloss=1):
    """
    Compute C such that C * Q_anchors = Q_bar minimized with Kullback-Leibler divergence.
    All rowsums of this matrix product are 1, for Q_* by construction, for C it follows.
    Params:
        Q: numpy float array, word coocurrence matrix
        anchors: list of indices of anchor words
        tol: tolerance for nmf
        beta_loss: 1 for Kullback-Leibler (more precise), 2 for L2 loss (faster)
    Returns:
        A: term x topic matrix
        C: intermediate result
        n_iter: number of iterations till convergence in computation of C
    """
    n_topics = len(anchors)
    P_w = Q.sum(axis=1)
    Q_bar = normalize(Q, axis=1, norm='l1')
    Q_anchors = Q_bar[anchors, :]

    #   Q_anchor
    # C Q_bar
    C, _, n_iter = non_negative_factorization(Q_bar, W=None, H=Q_anchors, n_components=n_topics,
                                              update_H=False, solver='mu', beta_loss=beta_loss,
                                              tol=tol)

    A_prime = np.multiply(P_w.reshape(-1, 1), C)
    A = normalize(A_prime, axis=0, norm='l1')

    return A, C, n_iter


class TopModel(BaseEstimator, TransformerMixin):
    """
    Nonnegative matrix factorization according to https://arxiv.org/pdf/1212.4777, where the
    term-topic-factor is explicitly constructed.

    Remark:
    The authors propose also faster versions using L2 norm as loss. Using `beta_loss=2`
    instead of 1 computes the L2 loss. But `C` is not row normalized then anymore. Is that ok?
    Or is some normalizatoin of C necessary before computing A? I think it is ok without extras.
    """
    def __init__(self, n_topics, anchor_thresh=100, proj_dim=2000, seed=None):
        """
        Params:
            doc_term_mat: scipy.sparse matrix as from CountVectorizer
            n_topics: number of topics
            anchor_thresh: number of documents an anchor word has to appear in (stability)
            seed: random seed for find_anchors
        """
        self.n_topics = n_topics
        self.anchor_thresh = anchor_thresh
        self.proj_dim = proj_dim
        self.seed = seed or int(time.time())

    def fit(self, doc_term_mat, tol=TOL, beta_loss=1):
        """
        Params:
            doc_term_mat: scipy.sparse matrix as from CountVectorizer
            tol: tolerance for nmf
            beta_loss: 1 for Kullback-Leibler (more precise), 2 for L2 loss (faster)
        """
        M = doc_term_mat.T.tocsc().astype(float)

        self.Q = get_Q(M)

        log.info("identifying candidate anchors")
        candidate_anchors = np.where((M > 0).sum(axis=1) > self.anchor_thresh)[0].tolist()

        self.anchors = find_anchors(
            self.Q, self.n_topics, candidate_anchors, self.proj_dim, self.seed)

        self.A, self.C, self.n_iter_fit = recover_term_topic_matrix(self.Q, self.anchors, tol=tol)

    def transform(self, doc_term_mat, tol=TOL, beta_loss=1):
        """
        Params:
            doc_term_mat: scipy.sparse matrix as from CountVectorizer
            tol: tolerance for nmf
            beta_loss: 1 for Kullback-Leibler (more precise), 2 for L2 loss (faster)
        Returns:
            W_T: topic x term matrix
        """
        M = doc_term_mat.T

        # product matrix M.T is document term matrix
        #      A.T
        # W.T  M.T
        W_T, _, self.n_iter_transform = non_negative_factorization(
            M.T, W=None, H=self.A.T, n_components=self.n_topics,
            update_H=False, solver='mu', beta_loss=beta_loss, tol=tol)

        return W_T

    @property
    def topic_term_matrix(self):
        return self.A.T.copy()
