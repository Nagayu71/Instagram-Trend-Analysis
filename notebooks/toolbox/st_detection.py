import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt



class STdetection():
    
    def __init__(self, G, tau, path):
        self.__G = G
        self.__tau = tau
        self.__aam = nx.to_numpy_array(self.__G)
        self.__n = len(self.__aam)
        self.__estimate = np.load(path)
        self.__nodelist = list(G.nodes())
    
    @property
    def tau(self):
        return self.__tau
    
    @property
    def aam(self):
        return self.__aam
    
    @property
    def n(self):
        return self.__n
    
    @property
    def estimate(self):
        return self.__estimate
    @property
    def nodelist(self):
        return self.__nodelist
    
    def connection_probability(self):

        p = (self.__estimate.reshape(-1,1)) @ (self.__estimate.reshape(1,-1))
        p = np.where(p < 1, p, 1)
        diag_ = np.diag_indices(self.__n)
        p[diag_] = 0
        assert np.min(p) >= 0 and np.max(p) <= 1, "Connection probability is not within the range."
        return p    

    
    def get_pvalues(self):
        prob = self.connection_probability()
        assert self.__n == len(prob), 'The size of the matrix AAM and the connection probability matrix are different.'

        # Initialize p-value (survival function)
        pval = np.ones((self.__n,self.__n))
        for i in range(self.__n):
            for j in range(self.__n):
                if i < j:
                    if self.__aam[i][j] > 0:
                        pval[i][j] = stats.binom.sf(k=self.__aam[i][j], n=self.__tau, p=prob[i][j]) 
        return pval
    
    def detect_st(self):
        # the number of edges to be tested
        N = np.sum(self.__aam > 0) / 2
        assert N == self.__G.number_of_edges(), 'calc went wrong'
        
        # Significance level corrected by Bonferroni correction
        sl_bf0 = 0.01 / N
        sl_bf1 = 0.001 / N
        
        # find p-values
        pv = self.get_pvalues()
        # adjacency matrix(2-D array) of significant ties
        st005 = np.where(pv < 0.05, 1, 0)
        st001 = np.where(pv < 0.01, 1, 0)
        st0001 = np.where(pv < 0.001, 1, 0)
        st_bf0 = np.where(pv < sl_bf0, 1, 0)
        st_bf1 = np.where(pv < sl_bf1, 1, 0)
        
        return st005, st001, st0001, st_bf0, st_bf1
    
    def show_ratio_ST(self, barWidth=0.925, ST_col='#69B3A2', Normal_col='#BDDDD5', anchor=(0.95, 0.95), loc='upper right', figsize=None, fname=None, *args):
        if args:
            # the number of significant ties
            str005 = np.sum(args[0])
            str001 = np.sum(args[1])
            str0001 = np.sum(args[2])
            str_bf0 = np.sum(args[3])
            str_bf1 = np.sum(args[4])
        else:
            # adjacency matrix of significant ties
            adj = self.detect_st()
            # the number of significant ties
            str005 = np.sum(adj[0])
            str001 = np.sum(adj[1])
            str0001 = np.sum(adj[2])
            str_bf0 = np.sum(adj[3])
            str_bf1 = np.sum(adj[4])
        
        # the number of edges to be tested
        N = np.sum(self.__aam > 0) / 2
        assert N == self.__G.number_of_edges(), 'aam does not correspond to the aggregate graph passed as argument.'
        sigTies = np.array([str005, str001, str0001, str_bf0, str_bf1])
        normalEdges = N - sigTies
        
        # find the ratio of significant ties to tested edges (STs) and that of normal ties (Normal)
        STs, Normal = sigTies / N, normalEdges / N
        
        # plot
        plt.rcParams['font.family'] = 'Arial'
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        names = (r'$\alpha = 0.05$',r'$\alpha = 0.01$',r'$\alpha = 0.001$','Bonferroni,\n'+r'$\alpha = 0.01$','Bonferroni,\n'+r'$\alpha = 0.001$')
        # Create Normal edges' Bars
        ax.bar(names, Normal, bottom=STs, color=Normal_col, edgecolor='white', width=barWidth, label="Normal edges")
        # Create Significant ties' Bars
        ax.bar(names, STs, color=ST_col, edgecolor='white', width=barWidth, label="Significant ties")
        # Custom x axis
        ax.set_xlabel("Significance level", fontsize = 14)
        ax.set_ylabel("Ratio of significant ties", fontsize = 14)
        ax.legend(loc=loc, bbox_to_anchor=anchor ,fontsize=12)
        fig.tight_layout()
        # Show graphic
        plt.show()
        
        if fname:
            fig.savefig(f"{fname}")
    

    def turn_graph(self, onlyst=True, fname=None):
        # Get adjacency matrix(2-D array) of significant ties at a Bonferroni-corrected significance level
        backbones = []
        for i in [2, 1]:
            STs = self.detect_st()[-i]
            N = len(STs) # original network size
            backbone = nx.from_pandas_adjacency(pd.DataFrame(STs, index=self.__nodelist, columns=self.__nodelist))
            assert N == backbone.number_of_nodes(), "The number of nodes does not match."

            if onlyst:
                backbone.remove_nodes_from(list(nx.isolates(backbone)))
            backbones.append(backbone)

            if fname:
                ref = {2: "B001", 1: "B0001"}
                nx.write_graphml(backbone, f'{fname}_{ref[i]}.graphml', encoding='utf-8')
        
        return backbones
