import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats

class AggregateNetwork():
    
    def __init__(self, G, tau, hashtag):

        self.__G = G
        self.__tau = tau
        self.__searchtag = hashtag
        self.__aam = nx.to_numpy_array(self.__G)
        self.__n = len(self.__aam)
    
    @property
    def tau(self):
        return self.__tau
    @property
    def searchtag(self):
        return self.__searchtag
    
    @property
    def aam(self):
        return self.__aam
    
    @property
    def n(self):
        return self.__n
    @property
    def G(self):
        return self.__G

    @G.setter
    def G(self, G):
        self.__G = G
    
    def check_degree(self, ascending=False):
        for node in self.__G.nodes():
            self.__G.nodes[node]['degree'] = self.__G.degree()[node]

        print(f'isolate:{list(nx.isolates(self.__G))}')

        df = pd.DataFrame.from_dict(dict(self.__G.nodes(data=True)), orient='index')

        if ascending:
            return df.sort_values(by='degree', ascending=True)
        else:
            return df.sort_values(by='degree', ascending=False)

    def plot_dd(self, bins, color='green'):
        degree = dict(self.__G.degree())
        plt.hist(degree.values(),
                bins = bins,
                lw=2,
                color=color,
                ec='white',
                density=False)
        plt.yscale('log')
        plt.xlabel("Degree", fontsize=20)
        plt.ylabel("# nodes", fontsize=20)
        pass
    
    def remove_searchtag(self):
        self.__G.remove_node(self.__searchtag)
        print(f"Isolates after removal of the searchtag: \n{list(nx.isolates(self.__G))}")
        print("These isolate nodes are to be removed.")
        self.__G.remove_nodes_from(list(nx.isolates(self.__G)))
        assert len(list(nx.isolates(self.__G))) == 0, "There is at least one isolate node left."
        print("============================================")
        print("The isolate nodes were successfully deleted.")


    def h(self, a):

        a = np.array(a)

        monomials = (self.__aam - self.__tau*(a.reshape(-1,1)@a.reshape(1,-1))) / (np.ones((n, n)) - (a.reshape(-1,1)@a.reshape(1,-1)))
        diag_ = np.diag_indices(self.__n)
        monomials[diag_] = 0

        poly = monomials.sum(axis=1)
        return poly


    def initial_value(self):
        return (self.__aam.sum(axis=1)/self.__tau) / np.sqrt(self.__aam.sum()/self.__tau)
    
    """
    the following is under development
    """

    def solve_h(self, init, method):
        """Returns a root of a vector function h

        Parameters
        ----------
        G : NetworkX graph
            The Networkx graph constructed from aggregate adjacency matrix.
        tau : int
            the number of snapshots
        Returns
        -------
        sol : SymPy OptimizeResult
        """
        return optimize.root(self.h, init, method)
    
    def get_pvalues(self, prob):
        assert self.__n == len(prob), 'The size of the matrix AAM and the connection probability matrix are different.'

        # survival function
        pval = np.ones((self.__n,self.__n))
        for i in range(self.__n):
            for j in range(self.__n):
                if self.__aam[i][j] > 0:
                    pval[i][j] = stats.binom.sf(k=self.__aam[i][j], n=self.__tau, p=prob[i][j]) 
        return pval
    