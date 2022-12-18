import numpy as np
import pandas as pd
import networkx as nx
import powerlaw
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import stats
from matplotlib.ticker import ScalarFormatter

__all__ = [
    "check_degree",
    "plot_dd",
    "plot_time_evolution",
    "fit_power_law",
    "compare_distribution",
    "plot_pdf_ccdf",
    "initial_value",
    "solve_h",
    "connection_probability",
    "show_opt_result",
    "save_opt_result",
    "plot_a_hat",
    "plot_a_hat_match",
    "get_pvalues",
    "detect_st",
    "show_ratio_ST",
    "generate_ST_graph",
]


def check_degree(G, ascending=False):
    """Returns pandas.DataFrame of degree with sorted values.
    
    Parameters
    ----------
    G : NetworkX graph
        The Networkx graph constructed from aggregate adjacency matrix.
    ascending : bool

    Returns
    -------
    Pandas DataFrame 
    """
    for node in G.nodes():
        G.nodes[node]['degree'] = G.degree()[node]

    print(f'singleton:{list(nx.isolates(G))}')

    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
    
    if ascending:
        return df.sort_values(by='degree', ascending=True)
    else:
        return df.sort_values(by='degree', ascending=False)


def plot_dd(G, bins, color='green'):
    """Plots the degree distribution of G.
    
    Parameters
    ----------
    G : NetworkX graph
        The Networkx graph constructed from aggregate adjacency matrix.
    bins : int
        default: square root of #nodes
    color : str
        default: 'green'
    """
    plt.rcParams['font.family'] = 'Arial'
    degree = dict(G.degree())
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


def plot_time_evolution(df, fname=None, ncolor="#FF7F0E", lcolor="#3399e6", linewidth=1.5):
    plt.rcParams['font.family'] = 'Arial'
    COLOR_NODE = ncolor
    COLOR_LINK = lcolor

    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.set_xlabel("Time steps", fontsize= 18)
    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    ax1.plot("tau", "N", data=df, color=COLOR_NODE, linewidth=linewidth, label=r"$N$"+" (left)")
    ax2.plot("tau", "L", data=df, color=COLOR_LINK, linewidth=linewidth,label=r"$L$"+" (right)", linestyle='--')

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower right', prop={"size": 14})

    ax1.set_ylabel(r"$N$", rotation="horizontal", fontsize= 20, y=0.45)
    ax1.tick_params(axis="y", labelcolor=COLOR_NODE, labelsize=12)
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax1.ticklabel_format(style="sci", axis="y",scilimits=(0,0))

    ax2.set_ylabel(r"$L$", rotation="horizontal", fontsize= 20)
    ax2.tick_params(axis="y", labelcolor=COLOR_LINK, labelsize=12)
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax2.ticklabel_format(style="sci", axis="y",scilimits=(0,0))
    fig.tight_layout()

    if fname:
        fig.savefig(f"{fname}")

def fit_power_law(G):
    """Returns the fitted degree exponent (alpha), its standard error (sigma), and the calculated optimal minimal degree (k_min).
    
    Parameters
    ----------
    G : NetworkX graph
        The network under investigation

    Returns
    -------
    (alpha, sigma, k_min) : tuple
        alpha : fitted degree exponent
        sigma : standard error
        k_min : calculated optimal minimal degree to start
    """
    plt.rcParams['font.family'] = 'Arial'
    data = list(dict(G.degree()).values())
    result = powerlaw.Fit(data)
    print('------------------------------------------------')
    print(f"the fitted parameter alpha: {result.power_law.alpha}, \nits standard error sigma: {result.power_law.sigma}, \nk_min: {result.power_law.xmin}")

    return result.power_law.alpha, result.power_law.sigma, result.power_law.xmin


def compare_distribution(G, dist1, dist2):
    """Returns more preferred distribution between candidate distributions.
    
    Parameters
    ----------
    G : NetworkX graph
        Network under investigation
    dist1 : string
        Name of the first candidate distribution (ex. ‘power_law’)
    dist2 : string
        Name of the second candidate distribution (ex. ‘exponential’)

    Returns
    -------
    Preferred distribution between candidates
    """

    data = list(dict(G.degree()).values())
    result = powerlaw.Fit(data)
    print('------------------------------------------------')

    R, p = result.distribution_compare(dist1, dist2)
    if R > 0:
        print(f"The {dist1} distribution outperforms {dist2} distribution with significance level {p}.")
    else:
        print(f"The {dist2} distribution outperforms {dist1} distribution with significance level {p}.")


def plot_pdf_ccdf(G, sep=False, Pcolor="#3399e6", Ccolor="#FF7F0E",fname=None):
    """
    1. Shows the fitted degree exponent (alpha), its standard error (sigma), and the calculated optimal minimal degree (k_min).
    2. Plots the probability density function (PDF) and the complementary cumulative distribution function (CCDF).
    
    Parameters
    ----------
    G : NetworkX graph
        The network under investigation

    Returns
    -------
    (alpha, sigma, k_min) : tuple
        alpha : fitted degree exponent
        sigma : standard error
        k_min : calculated optimal minimal degree to start
    """

    data = list(dict(G.degree()).values())
    result = powerlaw.Fit(data)
    print('------------------------------------------------')
    print(f"the fitted parameter alpha: {result.power_law.alpha}, \nits standard error sigma: {result.power_law.sigma}, \nk_min: {result.power_law.xmin}")
    
    plt.rcParams['font.family'] = 'Arial'
    COLOR_PDF = Pcolor
    COLOR_CCDF = Ccolor
    
    if sep:
        fig = plt.figure(figsize = (11,4))
        ax1 = fig.add_subplot(1, 2, 1)
        
        ax1 = result.plot_pdf(color=COLOR_PDF, linewidth=2.5, ax=ax1, label="PDF")
        result.power_law.plot_pdf(color=COLOR_PDF, alpha=0.6, linewidth=2, linestyle=':', ax=ax1, label="Theoretical PDF")
        ax1.set_xlabel(r"$k$", fontsize= 14)
        ax1.set_ylabel(r"$p(k)$", fontsize= 14)
        ax1.legend(loc='lower left',fontsize=12)
        
        ax2 = fig.add_subplot(1, 2, 2)
        result.plot_ccdf(color=COLOR_CCDF, linewidth=2.5, ax=ax2, label="CCDF")
        result.power_law.plot_ccdf(color=COLOR_CCDF, alpha=0.6, linewidth=2, linestyle=':', ax=ax2, label="Theoretical CCDF")
        ax2.set_xlabel(r"$k$", fontsize= 14)
        ax2.set_ylabel(r"$p(X\geq k)$", fontsize= 14, labelpad=2)
        ax2.legend(loc='lower left',fontsize=12)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax = result.plot_pdf(color=COLOR_PDF, linewidth=2.5, ax=ax, label="PDF")
        result.power_law.plot_pdf(color=COLOR_PDF, alpha=0.6, linewidth=2, linestyle=':', ax=ax, label="Theoretical PDF")
        result.plot_ccdf(color=COLOR_CCDF, linewidth=2.5, ax=ax, label="CCDF")
        result.power_law.plot_ccdf(color=COLOR_CCDF, alpha=0.6, linewidth=2, linestyle=':', ax=ax, label="Theoretical CCDF")
        ax.set_xlabel(r"$k$", fontsize= 14)
        ax.set_ylabel(r"$p(k),\;p(X\geq k)$", fontsize= 14)
        ax.legend(loc='lower left',fontsize=12)
        fig.tight_layout()
    if fname:
        fig.savefig(f"{fname}")


def initial_value(aam, tau):
    """Returns the initial values (guess) of nodes' activity levels.
    
    Parameters
    ----------
    aam : NumPy ndarray (2-D)
        ndarray of aggregate adjacency matrix
    tau : int
        the number of snapshots

    Returns
    -------
    NumPy ndarray : 1-D
    """
    return (aam.sum(axis=1)/tau) / np.sqrt(aam.sum()/tau)


def solve_h(G, tau, init, method="krylov"):
    """Returns a root of a vector function h.
    
    Parameters
    ----------
    G : NetworkX graph
        The Networkx graph constructed from aggregate adjacency matrix.
    tau : int
        the number of snapshots
    init : ndarray
        the initial value (guess) to start with
    method : str, optional
        Type of solver. "krylov" or "broyden1" is recommended.
        For more information, see the official documents:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html

    Returns
    -------
    sol : OptimizeResult
    """
    
    aam = nx.to_numpy_array(G)
    n = len(aam)
    
    def h(a):

        a = np.array(a)

        monomials = (aam - tau*(a.reshape(-1,1)@a.reshape(1,-1))) / (np.ones((n,n)) - (a.reshape(-1,1)@a.reshape(1,-1)))
        diag_ = np.diag_indices(n)
        monomials[diag_] = 0

        poly = monomials.sum(axis=1)
        return poly
    
    return optimize.root(h, init, method)


def connection_probability(sol_estimate):
    """Returns the connection probabilities between nodes in matrix form.

    Parameters
    ----------
    sol_estimate：x property of OptimizeResult

    Returns
    -------
    NumPy ndarray : 2-D
        Each entry is connection probability of node i and node j.
        Diagonal component is always 0.
    """
    p = (sol_estimate.reshape(-1,1)) @ (sol_estimate.reshape(1,-1))
    p = np.where(p < 1, p, 1)
    n = len(p)
    diag_ = np.diag_indices(n)
    p[diag_] = 0
    assert np.min(p) >= 0 and np.max(p) <= 1, "Connection probability is not within the range."
    return p    


def show_opt_result(sol, method):
    """Returns basic information about the optimization outcome.
    
    Parameters
    ----------
    sol：OptimizeResult
    method : str, optional
        Type of solver.
    """
    print(f'{method}')
    print(f'success? : {sol.success}')
    print(f'estimate : {sol.x}')
    # check missing value
    print(f'missing value : {np.isnan(sol.x).sum()}')
    # see if 0 < a <= 1
    print(f'min : {np.min(sol.x)}')
    print(f'max : {np.max(sol.x)}')
    print('======================================')
    p = (sol.x.reshape(-1,1)) @ (sol.x.reshape(1,-1))
    n = len(p)
    diag_ = np.diag_indices(n)
    p[diag_] = 0
    print(f'maximum connection Prob u(a_i,a_j):{np.max(p)}')
    print(f'index of maximum value : {np.unravel_index(np.argmax(p), p.shape)}')


def save_opt_result(sol, fname=None):
    """Save an array of the optimization outcome to a binary file in NumPy ``.npy`` format.
    
    Parameters
    ----------
    sol：OptimizeResult
    fname : str, optional
        Filename to which the array is saved.
    """
    np.save(f'{fname}', sol.x)


def plot_a_hat(sol_krylov, sol_broyden, cmap="cool", color="c",alpha=0.3, figsize=(12,4), sep=True, fname=None):
    """Plots the estimates of activity levels estimated by Krylov approximation and Broyden good method.
    If a file name is passed as an argument, the output figure is saved in a user-defined data format.

    Parameters
    ----------
    sol_krylov：OptimizeResult
        The solution represented as a ``OptimizeResult`` object.
    sol_broyden：OptimizeResult
        The solution represented as a ``OptimizeResult`` object.
    cmap : str or Colormap (default: rcParams["image.cmap"])
        The Colormap instance or registered colormap name used to map scalar data to colors.
    fname : str, optional
        Filename to which the figure is saved. Be sure to add a file extension.
        If fname has no extension, then the file is saved as png in default.
    """
    plt.rcParams['font.family'] = 'Arial'

    if sep:
        fig = plt.figure(figsize = figsize)

        ax0 = fig.add_subplot(121)
        ax0.scatter(np.arange(1,len(sol_krylov.x)+1), np.sort(sol_krylov.x), s = 4 , c = np.sort(sol_krylov.x), cmap=cmap, alpha=alpha)
        ax0.set_xlabel("Krylov", fontsize=14)
        ax0.set_ylabel("Activity level", fontsize=14)
        ax0.set_yscale("log")

        ax1 = fig.add_subplot(122)
        ax1.scatter(np.arange(1,len(sol_broyden.x)+1), np.sort(sol_broyden.x), s = 4 , c = np.sort(sol_broyden.x), cmap=cmap, alpha=alpha)
        ax1.set_xlabel("Broyden’s good method", fontsize=14)
        ax1.set_yscale("log")

        fig.tight_layout()
    else:
        fig, ax = plt.subplots()
        ax.scatter(sol_krylov.x, sol_broyden.x, linewidths=0.5, marker="+", c=color, alpha=alpha, label='Activity level')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Krylov", fontsize=14)
        ax.set_ylabel("Broyden’s good method", fontsize=14)
        ax.legend(loc='lower right', fontsize=12)
        fig.tight_layout()
    
    if fname:
        fig.savefig(f"{fname}")


def plot_a_hat_match(sol_default, sol_another, xlabel, ylabel, marker="+", color="c", figsize=None, fname=None):
    """Plots the estimates of activity levels estimated by Krylov approximation and Broyden good method.
    If a file name is passed as an argument, the output figure is saved in a user-defined data format.

    Parameters
    ----------
    sol_default：OptimizeResult
        The solution represented as a ``OptimizeResult`` object.
    sol_another：OptimizeResult
        The solution represented as a ``OptimizeResult`` object.
    cmap : str or Colormap (default: rcParams["image.cmap"])
        The Colormap instance or registered colormap name used to map scalar data to colors.
    fname : str, optional
        Filename to which the figure is saved. Be sure to add a file extension.
        If fname has no extension, then the file is saved as png in default.
    """
    plt.rcParams['font.family'] = 'Arial'

    fig, ax = plt.subplots()
    ax.scatter(sol_default.x, sol_another.x, linewidths=0.5, marker=marker, c=color, label='Activity level')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc='lower right', fontsize=12)
    fig.tight_layout()
    
    if fname:
        fig.savefig(f"{fname}")


def get_pvalues(aam, tau, prob):
    """Returns the P-values of the test obtained from the actual empirical number of interactions between nodes.
    
    Parameters
    ----------
    aam : NumPy ndarray (2-D)
        adjacency matrix of aggregate network (aggregate adjacency matrix)
    tau : int
        the number of snapshots
    prob : NumPy ndarray (2-D)
        connection probability matrix

    Returns
    -------
    NumPy ndarray : 2-D
        Each element of the upper triangular matrix represents the p-value of each edge to be tested.
        All elements except the upper triangular matrix are 1.
    """
    n = len(aam)
    assert n == len(prob), 'The size of the aggregate adjacency matrix and the connection probability matrix are different.'
    
    # survival function
    pval = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            if i < j:
                if aam[i][j] > 0:
                    pval[i][j] = stats.binom.sf(k=aam[i][j], n=tau, p=prob[i][j]) 
    
    assert np.min(pval) > 0 and np.max(pval) <= 1, "The p-value is not within the range."
    return pval

def detect_st(G, aam, tau, prob, pvalue=None):
    """Returns the adjacency matrix of significant ties as a NumPy array.
    
    Parameters
    ----------
    G : NetworkX graph
        The Networkx graph constructed from aggregate adjacency matrix.
    aam : NumPy ndarray (2-D)
        adjacency matrix of aggregate network (aggregate adjacency matrix)
    tau : int
        the number of snapshots
    prob : NumPy ndarray (2-D)
        connection probability matrix
    pvalue : NumPy ndarray (2-D), optional

    Returns
    -------
    Tuple consisting of four adjacency matrices each of which is obtained by respective significance level.
    Each matrix is given as a NumPy ndarray (2-D).
    """
    # the number of edges to be tested
    N = np.sum(aam > 0) / 2
    assert N == G.number_of_edges(), 'aam does not correspond to the aggregate graph passed as argument.'

    # Significance level corrected by Bonferroni correction
    sl_bf = 0.01 / N
    
    if pvalue:
        # adjacency matrix(2-D array) of significant ties
        st005 = np.where(pvalue < 0.05, 1, 0)
        st001 = np.where(pvalue < 0.01, 1, 0)
        st0001 = np.where(pvalue < 0.001, 1, 0)
        st_bf = np.where(pvalue < sl_bf, 1, 0)
    else:
        # find p-values
        pv = get_pvalues(aam, tau, prob)
        # adjacency matrix(2-D array) of significant ties
        st005 = np.where(pv < 0.05, 1, 0)
        st001 = np.where(pv < 0.01, 1, 0)
        st0001 = np.where(pv < 0.001, 1, 0)
        st_bf = np.where(pv < sl_bf, 1, 0)

    return st005, st001, st0001, st_bf

def show_ratio_ST(G, aam, tau, prob, pvalue=None, fname=None, *args):
    """Plots the ratio of significant ties to the whole tested edges.
    If a file name is passed as an argument, the output figure is saved in a user-defined data format.

    Parameters
    ----------
    G : NetworkX graph
        The Networkx graph constructed from aggregate adjacency matrix.
    aam : NumPy ndarray (2-D)
        adjacency matrix of aggregate network (aggregate adjacency matrix)
    tau : int
        the number of snapshots
    prob : NumPy ndarray (2-D)
        connection probability matrix
    pvalue : NumPy ndarray (2-D), optional
    fname : str, optional
        Filename to which the figure is saved. Be sure to add a file extension.
        If fname has no extension, then the file is saved as png in default.
    args : optional
        You can pass four adjacency matrices of significant ties arranged 
        in order of significance level: 0.05, 0.01, 0.001, Bonferroni.
    """

    plt.rcParams['font.family'] = 'Arial'

    if args:
        # the number of significant ties
        str005 = np.sum(args[0])
        str001 = np.sum(args[1])
        str0001 = np.sum(args[2])
        str_bf = np.sum(args[3])
    else:
        # adjacency matrix of significant ties
        adj = detect_st(G, aam, tau, prob, pvalue=None)
        # the number of significant ties
        str005 = np.sum(adj[0])
        str001 = np.sum(adj[1])
        str0001 = np.sum(adj[2])
        str_bf = np.sum(adj[3])

    # the number of edges to be tested
    N = np.sum(aam > 0) / 2
    assert N == G.number_of_edges(), 'aam does not correspond to the aggregate graph passed as argument.'
    sigTies = np.array([str005, str001, str0001, str_bf])
    normalEdges = N - sigTies

    # find the ratio of sinnificant ties to tested edges (greenbar) and that of normal ties (orangebar)
    greenBars, orangeBars = sigTies / N, normalEdges / N

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()
    barWidth = 0.85
    names = (r'$\alpha = 0.05$',r'$\alpha = 0.01$',r'$\alpha = 0.001$',r'Bonferroni, $\alpha = 0.01$')
    # Create green Bars
    ax.bar(names, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Significant ties")
    # Create orange Bars
    ax.bar(names, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth, label="Normal edges")
    # Custom x axis
    ax.set_xlabel("Significance level", fontsize = 14)
    ax.set_ylabel("Ratio of significant ties", fontsize = 14)
    ax.legend(loc='lower left', bbox_to_anchor=(0.02,0.03) ,fontsize=12)
    # Show graphic
    plt.show()

    if fname:
        fig.savefig(f"{fname}")


def generate_ST_graph(G, aam, tau, prob, pvalue=None, onlyst=True, fname=None):
    """Returns a graph in which only significant ties at a Bonferroni-corrected significance level are linked.
    If a file name is passed as an argument, the graph is saved in graphml format.

    Parameters
    ----------
    G : NetworkX graph
        The Networkx graph constructed from aggregate adjacency matrix.
    aam : NumPy ndarray (2-D)
        adjacency matrix of aggregate network (aggregate adjacency matrix)
    tau : int
        the number of snapshots
    prob : NumPy ndarray (2-D)
        connection probability matrix
    pvalue : NumPy ndarray (2-D), optional
    fname : str, optional
        File or filename to write. Filenames ending in .gz or .bz2 will be compressed.
    onlyst : bool
        default
    Returns
    -------
    Tuple consisting of four adjacency matrices each of which is obtained by respective significance level.
    Each matrix is given as a NumPy ndarray (2-D).
    """
    # Get adjacency matrix(2-D array) of significant ties at a Bonferroni-corrected significance level
    STs = detect_st(G, aam, tau, prob, pvalue=None)[-1]
    nodes = list(G.nodes())
    backbone = nx.from_pandas_adjacency(pd.DataFrame(STs, index=nodes, columns=nodes))

    if onlyst:
        backbone.remove_nodes_from(list(nx.isolates(backbone)))

    if fname:
        nx.write_graphml(backbone, f'{fname}.graphml', encoding='utf-8')
    
    return backbone