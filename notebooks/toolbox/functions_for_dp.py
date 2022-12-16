import re
import os
import json
import pandas as pd
from datetime import timedelta
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from collections import Counter
import itertools


__all__ = [
    "merge_raw_data",
    "extract_hashtag",
    "get_dataframe",
    "get_snapshots_closed_intervals",
    "get_each_NL",
    "get_flattened_temporally_aggregated_network",
    "evolution_of_network",
    "get_edgelist_of_eachsnapshot",
    "make_aggregate_adjacency_matrix",
]


def merge_raw_data(querys, timespan, save=False):
    for q in querys:
        # ディレクトリ内のファイル名をリストで取得
        #dir_path = f"../data/{q}"
        dir_path = f"../data/{q}"
        files = os.listdir(dir_path)

        # pd.DataFrame()に渡すリストを作成
        pd_list = []

        # 最新の投稿から降順で表示するために`files`からファイルを逆順に取り出す
        for file in reversed(files):
            with open(f'{dir_path}/{file}', encoding='utf-8') as f:
                opened_data = f.read()
                loaded_data = json.loads(opened_data)
                pd_list.extend(loaded_data)

        # create dataframe
        df = pd.DataFrame(pd_list)
        assert len(pd_list) == len(df), "The number of data contradicts."

        # Extract necessary rows
        df = df.loc[:,['timestamp', 'id', 'caption']]

        # UTC-->JST
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert('Asia/Tokyo')

        # Set timestamp as index
        df = df.set_index('timestamp')
        
        # 重複していない行の数
        unique = df.duplicated(subset='id').value_counts()[0]

        # 重複した行を落として、新しいデータフレームを作成
        unique_df = df.drop_duplicates(subset='id')
        #assert unique == len(unique_df), "There is a problem in data merging process."

        file_name = f"./data/datasets/{q}/{q}_{timespan}.pkl"
        
        print(f"unique rows : {len(unique_df)}", file_name)
        
        # pickleファイルで出力
        if save:
            unique_df.to_pickle(file_name)


def extract_hashtag(caption):
    try:
        pattern = r'#([^#\s$]+)'
        result = re.findall(pattern, caption)
        return result
    except:
        return []


def get_dataframe(hashtag, file):
    # read row data
    df = pd.read_pickle(file)
    
    # Extract hashtags in caption and make a new series for them
    df['hashtags'] = df['caption'].apply(extract_hashtag)
    
    # If missing any hashtag in the caption, complement with the hashtag used for search.
    for lst in df["hashtags"]:
        if hashtag in lst:
            pass
        else:
            lst.append(hashtag)


    # Create the edge lists to generate a graph
    edge_lists = []
    for post_id, hashtags in zip(df.id, df.hashtags):
        edge_list = []
        for ht in hashtags:
            edge_list.append((post_id, ht))
        edge_lists.append(edge_list)

    df['edge_list'] = edge_lists

    # Check to see if the edge list was created correctly
    for tags, edges in zip(df['hashtags'], df['edge_list']):
        assert len(tags) == len(edges), "Fail to create edge list correctly."
    
    return df


def get_snapshots_closed_intervals(df, delta='minutes=30'):
    """ Calculate tau (# snapshots) and make a list of the closed interval of each snapshot
    
    Parameters
    ----------
    df : pandas.DataFrame
        the whole data to analyze
    delta : str
        temporal resolution (default: 'minutes=30', format: 'minutes=0'/'hours=0')

    Returns
    -------
    tuple
    - tau (int)
    - list of the closed interval of each snapshot
    """
    #pd_freq = ['min', 'H']
    
    if 'minutes' in delta:
        freq = delta.split('=')[1] + 'min'
        tdelta = timedelta(minutes=int(delta.split('=')[1]))
    else:
        freq = delta.split('=')[1] + 'H'
        tdelta = timedelta(hours=int(delta.split('=')[1]))
    
    # timestamps are in descending order
    timestamps = list(df.index)
    # Get the whole data temporal window of length
    T = timestamps[0] - timestamps[-1]
    # Compute tau
    result = T / tdelta # float型
    tau = np.floor(result)
    
    interval_list = list(pd.date_range(timestamps[-1], timestamps[0],freq=freq,inclusive='left'))
    assert tau == len(interval_list)-1, f"tau {(tau)} and #snapshots {(len(interval_list)-1)} do not much."

    snapshots = list(itertools.pairwise(interval_list))
    assert len(snapshots) == tau, f"The number of snapshots {(len(snapshots))} and the value of tau {(tau)} do not much."
    
    return int(tau), snapshots


def get_each_NL(df, tau, snapshots):
    """Returns the number of nodes (N) and edges (L) in the graph of each snapshot.
    
    Parameters
    ----------
    df : pandas.DataFrame
        the whole data to analyze
    tau : int
        the number of snapshots
    snapshots : list of the closed interval of each snapshot

    Returns
    -------
    (N, L) : tuple
    """
    
    snapshots_idx = []
    snapshot_property = [] # the list in which to save (N, L)
    count = 0
    # Iterate over snapshots and make the tuple (N, L) of each snapshot
    for delta in snapshots:
        count += 1
        # Extract dataframe within the range of delta (tuple)
        df_delta = df[(delta[0].to_pydatetime() <= df.index) & (df.index < delta[1].to_pydatetime())]

        # Create edgelist for bipartite graph
        edge_list = []
        for lst in df_delta['edge_list']:
            edge_list.extend(lst)

        # Generate graph object from edgelist
        B = nx.Graph(edge_list)
        top_nodes = set(s for s,t in edge_list) # post id
        bottom_nodes = set(B) - top_nodes # hashtags

        # Project graph B onto hashtag node sets setting the elements of the adjacency matrix to 1
        hashtag_graph = bipartite.projected_graph(B, bottom_nodes)

        # Append index of snapshot to snapshots_idx
        snapshots_idx.append(f'snapshot{count}')
        
        # Calculate #nodes and #edges and save them in tuple
        snapshot_prop = (hashtag_graph.number_of_nodes(), hashtag_graph.number_of_edges())
        assert len(hashtag_graph.edges()) == snapshot_prop[1], "Failure: calculation of the number of edges in network."
        
        # Append each snapshot's property in snapshot_property
        snapshot_property.append(snapshot_prop)

    assert len(snapshots_idx) == tau, "failed to decompose DataFrame into tau snapshots."
    
    return dict(zip(snapshots_idx, snapshot_property))


def get_flattened_temporally_aggregated_network(df, snapshots):
    # Extract dataframe within the whole time span T
    df_delta = df[(snapshots[0][0].to_pydatetime() <= df.index) & (df.index <= snapshots[-1][-1].to_pydatetime())]
    # Create edgelist for bipartite graph
    edge_list = []
    for lst in df_delta['edge_list']:
        edge_list.extend(lst)

    # Generate graph object from edgelist
    B = nx.Graph(edge_list)
    top_nodes = set(s for s,t in edge_list) # post id
    bottom_nodes = set(B) - top_nodes # hashtags

    # Project graph B onto hashtag node sets setting the elements of the adjacency matrix to 1
    hashtag_graph = bipartite.projected_graph(B, bottom_nodes)

    return hashtag_graph


def evolution_of_network(df, tau, snapshots):
    """Returns the evolution of network property (N, L).
    
    Parameters
    ----------
    df : pandas.DataFrame
        the whole data to analyze
    tau : int
        the number of snapshots
    snapshots : list of the closed interval of each snapshot
    
    Returns
    -------
    (N, L) : tuple
        N : time-evolving number of nodes (List)
        L : time-evolving number of edges (List)
    """
    snapshots_idx = []
    time_evolving_N = []
    time_evolving_L = []
    count = 0
    # Iterate over snapshots and make the tuple (N, L) of time-evolving network
    for delta in snapshots:
        count += 1
        # Append index of snapshot to snapshots_idx
        snapshots_idx.append(f'snapshot{count}')
        
        # Extract dataframe within the range
        df_delta = df[(snapshots[0][0].to_pydatetime() <= df.index) & (df.index < delta[1].to_pydatetime())]

        # Create edgelist for bipartite graph
        edge_list = []
        for lst in df_delta['edge_list']:
            edge_list.extend(lst)

        # Generate graph object from edgelist
        B = nx.Graph(edge_list)
        top_nodes = set(s for s,t in edge_list) # post id
        bottom_nodes = set(B) - top_nodes # hashtags

        # Project graph B onto hashtag node sets setting the elements of the adjacency matrix to 1
        hashtag_graph = bipartite.projected_graph(B, bottom_nodes)

        # Calculate #nodes and #edges and save them in time-evolving lists respectively
        time_evolving_N.append(hashtag_graph.number_of_nodes())
        time_evolving_L.append(hashtag_graph.number_of_edges())

    assert len(snapshots_idx) == tau, "failed to decompose DataFrame into tau snapshots."
    
    return time_evolving_N, time_evolving_L


def get_edgelist_of_eachsnapshot(df, tau, snapshots, debug=True):
    """
    デバッグのためにタプルで使用する変数を返している
    """
    snapshots_idx = []
    T_edge_2dlists = [] # 各スナップショットに含まれるエッジリストを格納する２次元リスト
    T_edge_count = 0
    # Iterate over snapshots and make the edge list of the snapshot with snapshot's id
    count = 0
    for delta in snapshots:
        count += 1
        # delta (tuple) のデータフレームを抜き出し
        df_delta = df[(delta[0].to_pydatetime() <= df.index) & (df.index < delta[1].to_pydatetime())]

        # 2部グラフ用エッジ一リストの作成
        edge_list = []
        for lst in df_delta['edge_list']:
            edge_list.extend(lst)

        # エッジリストからグラフオブジェクトを生成
        B = nx.Graph(edge_list)
        top_nodes = set(s for s,t in edge_list) # post id
        bottom_nodes = set(B) - top_nodes # hashtags

        # 隣接行列の要素を1にするために、重みなしでプロジェクション
        hashtag_graph = bipartite.projected_graph(B, bottom_nodes)

        # snapshotのインデックスを格納
        snapshots_idx.append(f'snapshot{count}')
        # T_edge_listsにsnapshotごとのエッジリストを格納
        T_edge_2dlists.append(list(hashtag_graph.edges()))

        # T_edge_countにエッジリストの累計を記録（重複許す）
        T_edge_count += len(hashtag_graph.edges())

    assert len(T_edge_2dlists) == tau, "failed to aggregate edge_list."

    if debug:
        return T_edge_count, T_edge_2dlists, dict(zip(snapshots_idx, T_edge_2dlists))
    else:
        return dict(zip(snapshots_idx, T_edge_2dlists))


def make_aggregate_adjacency_matrix(edgelst_of_snapshots, fname=None):
    T_edge_lists = list(itertools.chain.from_iterable(edgelst_of_snapshots[1]))
    assert len(T_edge_lists) == edgelst_of_snapshots[0], "failed to flatten T_edge_2dlists"
    
    # Find the number of edge occurrences between node i and node j: m_ij
    # key is edge (node_i, node_j), value is the number of occurrences
    edge_occur = dict(Counter(T_edge_lists))
    
    # Turn edge_occur into an edge list for networkx
    edgelist = []
    for k,v in edge_occur.items():
        edgelist.append((k[0], k[1], {'weight': v}))
    
    # Create a graph from the edge list
    G = nx.Graph(edgelist)
    
    if fname:
        nx.write_graphml(G, f'{fname}.graphml', encoding='utf-8')
    
    return G