import time

import networkx as nx
import numpy as np
import scipy as sp
import scipy.sparse as sps
from sklearn import preprocessing
from sklearn.utils.extmath import randomized_svd


class SparseMatrixFactorization():
    def __init__(self, graph, dimension):
        self.g = graph
        self.dimension = dimension

        self.number_of_nodes = self.g.number_of_nodes()
        self.matrix = nx.adjacency_matrix(self.g)

    def get_embedding_rand(self, matrix):
        start_time = time.time()
        length = matrix.shape[0]
        sparse_matrix = sps.csc_matrix(matrix)

        U, sigma, VT = randomized_svd(sparse_matrix, n_components=self.dimension, n_iter=5, random_state=None)
        U = U * np.sqrt(sigma)
        U = preprocessing.normalize(U, 'l2')

        return U

    def pre_factorization(self, tran, mask):
        start_time = time.time()
        l1 = 0.75
        C1 = preprocessing.normalize(tran, 'l1')
        neg = np.array(C1.sum(axis=0))[0] ** l1

        neg = neg / neg.sum()

        neg = sps.diags(neg, format='csr')
        neg = mask.dot(neg)

        C1.data[C1.data <= 0] = 1
        neg.data[neg.data <= 0] = 1

        C1.data = np.log(C1.data)
        neg.data = np.log(neg.data)

        C1 -= neg
        F = C1

        f_matrix = self.get_embedding_rand(F)

        return f_matrix
