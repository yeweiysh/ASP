import scipy.io as sio
import numpy as np
import networkx as nx
import igraph as ig
import ot
import re
from networkx.algorithms.traversal import breadth_first_search as bfs

from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.base import clone
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.svm import SVC

from tqdm import tqdm

def load_data(dataset):
    data = sio.loadmat('./dataset/%s.mat' % dataset)
    graphs_data = data['graph']
    graphs_num = len(graphs_data[0])

    return graphs_data, graphs_num


def get_graph_label(dataset):
    data = sio.loadmat('./dataset/%s.mat' % dataset)
    graph_label = [int(label) for label in data['label']]

    return np.array(graph_label)


def get_node_label(graph_datas, graphs_num):
    labels = {}
    for gidx in range(graphs_num):
        label = graph_datas[0][gidx]['nl'].T
        labels[gidx] = label[0]

    return labels

def jaccard_distance(neighbors_u, neighbors_v):
    intersection = list(set(neighbors_u).intersection(set(neighbors_v)))
    union = list(set(neighbors_u).union(set(neighbors_v)))

    return 1 - (len(intersection)) / (len(union))


def preprocess_graphs(G, graphs_num):
    graphs_data = []
    for i in range(graphs_num):
        adj_matrix = G[0][i]['am']
        nx_G = nx.from_numpy_matrix(adj_matrix)
        ig_G = ig.Graph.from_networkx(nx_G)

        node_label = G[0][i]['nl'].squeeze()
        ig_G.vs['label'] = node_label

        all_neighbors = []
        for j in range(ig_G.vcount()):
            neighbors = ig_G.neighbors(j)
            neighbors.insert(0, j)
            all_neighbors.append(neighbors)

        for idx, edge in enumerate(ig_G.es):
            u, v = edge.source, edge.target
            weight = jaccard_distance(all_neighbors[u], all_neighbors[v])
            ig_G.es[idx]['weight'] = weight

        graphs_data.append(ig_G)

    return graphs_data


def filtration_by_edge_attribute(graph, delete_nodes=False, stop_early=False):
    weights = np.array(graph.es['weight'])
    # print(weights)

    F = []

    # node_num = graph.vcount()

    if weights.size != 1:
        weights = weights
        x = False
    else:
        weights = np.array([weights])
        x = True

    # print(x)

    for weight in sorted(weights):
        if x:
            weight = weight[0]
        edges = graph.es.select(lambda edge: edge['weight'] <= weight)
        subgraph = edges.subgraph(delete_vertices=delete_nodes)

        F.append((weight, subgraph))

        if stop_early and subgraph.vcount() == graph.vcount():
            break
    # print(F)
    graph_dict = {val[0]: val[1] for val in F}
    # print(graph_dict)
    return [G for G in graph_dict.values()]
    # return F


def get_node_idx(filtrated_graph):
    source_idx = [edge.source for edge in filtrated_graph.es]
    target_idx = [edge.target for edge in filtrated_graph.es]
    node_idx = list(set(source_idx).union(set(target_idx)))

    return node_idx


def ig_to_nx(graph):
    adj_matrix = [row for row in graph.get_adjacency()]
    nx_G = nx.from_numpy_array(np.array(adj_matrix))

    return nx_G


def compute_wasserstein_distance_matrix(embeddings):
    n = len(embeddings)
    M = np.zeros((n, n))
    # with tqdm(total=n) as p_bar:
    for graph_index_1, graph_1 in enumerate(embeddings):
        embedding_1 = embeddings[graph_index_1]
        for graph_index_2, graph_2 in enumerate(embeddings[graph_index_1:]):
            embedding_2 = embeddings[graph_index_2 + graph_index_1]
            ground_distance = 'euclidean'
            # ground_distance = 'hamming'
            cost = ot.dist(embedding_1, embedding_2, ground_distance)
            M[graph_index_1, graph_index_2 +
                graph_index_1] = ot.emd2([], [], cost)
            # p_bar.update(1)

    M = M + M.T

    return M


def node_relabeling(graph_datas, graphs_num, alllabels, maxh):
    for deep in range(1, maxh):
        # print(deep)
        labeledtrees = []
        labels_set = set()
        labels = {}
        labels = alllabels[0]
        for gidx in range(graphs_num):
            adj_matrix = graph_datas[0][gidx]['am']
            nx_G = nx.from_numpy_matrix(adj_matrix)
            label = labels[gidx]
            for node in range(len(nx_G)):
                edges = list(bfs.bfs_edges(nx_G, source=node, depth_limit=deep))
                bfstree = ''
                cnt = 0
                for u, v in edges:
                    bfstree += str(label[int(u)])
                    bfstree += ','
                    bfstree += str(label[int(v)])
                    if cnt < len(list(edges)):
                        bfstree += ','
                    cnt += 1
                labeledtrees.append(bfstree)
                labels_set.add(bfstree)
        labels_set = list(labels_set)
        labels_set = sorted(labels_set)

        index = 0
        labels = {}
        for gidx in range(graphs_num):
            adj_matrix = graph_datas[0][gidx]['am']
            n = len(adj_matrix)
            label = np.zeros(n)
            for node in range(n):
                label[node] = labels_set.index(labeledtrees[node+index])
            index += n
            labels[gidx] = label
        alllabels[deep] = labels

    return alllabels


def path_pattern_generation(nx_G, depth, node_idx=None):
    # if node_idx:
    #     node_range = node_idx
    # else:
    #     node_range = range(len(nx_G))
    node_range = node_idx if node_idx else range(len(nx_G))
    # print(node_range)

    paths_graph = []
    judge_set = set()
    # for node in range(len(nx_G)):
    for node in node_range:
        # print(node)
        paths_graph.append(str(node))
        judge_set.add(str(node))
        edges = list(bfs.bfs_edges(nx_G, source=node, depth_limit=depth))
        node_in_path = []
        for u, v in edges:
            node_in_path.append(v)
        pathss = []
        for i in range(len(edges)):
            path = list(nx.shortest_path(nx_G, node, node_in_path[i]))
            strpath = ''
            cnt = 0
            for vertex in path:
                cnt += 1
                strpath += str(vertex)
                if cnt < len(path):
                    strpath += ','
            pathss.append((strpath))

        for path in pathss:
            vertices = re.split(',', path)  # remove all ','
            rvertices = list(reversed(vertices))
            rpath = ''
            cnt = 0
            for rv in rvertices:
                cnt += 1
                rpath += rv
                if cnt < len(rvertices):
                    rpath += ','
            # remove repetitive path patterns. e.g '1,4' is equal to '4,1'
            if rpath not in judge_set:
                judge_set.add(path)
                paths_graph.append(path)
            else:
                paths_graph.append(rpath)

    return paths_graph


def path_pattern_labeling(paths, label):
    tmp_labeledpaths = []
    for path in paths:
        labeledpaths = ''
        vertices = re.split(',', path)
        cnt = 0
        for vertex in vertices:
            cnt += 1
            labeledpaths += str(int(label[int(vertex)]))
            if cnt < len(vertices):
                labeledpaths += ','
        tmp_labeledpaths.append(labeledpaths)

    return tmp_labeledpaths


def custom_grid_search_cv(model, param_grid, precomputed_kernels, y, cv=5):
    cv = StratifiedKFold(n_splits=cv, shuffle=False)
    results = []
    for train_index, test_index in cv.split(precomputed_kernels[0], y):
        split_results = []
        params = []
        for K_idx, K in enumerate(precomputed_kernels):
            for p in list(ParameterGrid(param_grid)):
                sc = _fit_and_score(clone(model), K, y, scorer=make_scorer(accuracy_score), train=train_index, test=test_index, verbose=0, parameters=p, fit_params=None)
                split_results.append(sc.get('test_scores'))
                params.append({'K_idx': K_idx, 'params': p})
        results.append(split_results)
    results = np.array(results)
    fin_results = results.mean(axis=0)
    best_idx = np.argmax(fin_results)
    ret_model = clone(model).set_params(**params[best_idx]['params'])
    return ret_model.fit(precomputed_kernels[params[best_idx]['K_idx']], y), params[best_idx]


def run_classification(distance_matrix, y, gridsearch, crossvalidation):
    # M1 = compute_wasserstein_distance_matrix(graph_embeddings)

    if gridsearch:
        gammas = np.logspace(-4, 1, num=6)
        # gammas = [0.0001]
        param_grid = [{'C': np.logspace(-3, 3, num=7)}]
    else:
        gammas = [0.001]

    kernel_matrices = []
    kernel_params = []

    for ga in gammas:
        K = np.exp(-ga*distance_matrix)
        kernel_matrices.append(K)
        kernel_params.append(ga)

    accuracy_scores = []
    np.random.seed(42)

    cv = StratifiedKFold(n_splits=10, shuffle=True)
    best_C = []
    best_gamma = []
    for train_index, test_index in cv.split(kernel_matrices[0], y):
        K_train = [K[train_index][:, train_index] for K in kernel_matrices]
        K_test = [K[test_index][:, train_index] for K in kernel_matrices]
        y_train, y_test = y[train_index], y[test_index]
        if gridsearch:
            gs, best_params = custom_grid_search_cv(SVC(kernel='precomputed'), param_grid, K_train, y_train, cv=5)
            C_ = best_params['params']['C']
            gamma_ = kernel_params[best_params['K_idx']]
            y_pred = gs.predict(K_test[best_params['K_idx']])
        else:
            gs = SVC(C=100, kernel='precomputed').fit(K_train[0], y_train)
            y_pred = gs.predict(K_test[0])
            gamma_, C_ = gammas[0], 100
        best_C.append(C_)
        best_gamma.append(gamma_)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        if not crossvalidation:
            break

    if crossvalidation:
        print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(np.mean(accuracy_scores) * 100, np.std(accuracy_scores) * 100))
    else:
        print('Final accuracy: {:2.3f} %'.format(np.mean(accuracy_scores) * 100))