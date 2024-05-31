import utils
import numpy as np
import networkx as nx
import sys
import time
from gensim import corpora
import gensim


if __name__ == '__main__':
    dataset = sys.argv[1]
    maxh = int(sys.argv[2])
    depth = int(sys.argv[3])

    graph_datas, graphs_num = utils.load_data(dataset)
    weighted_graph_datas = utils.preprocess_graphs(graph_datas, graphs_num)

    time_start = time.time()

    alllabels = {}
    labels = utils.get_node_label(graph_datas, graphs_num)
    alllabels[0] = labels

    alllabels = utils.node_relabeling(graph_datas, graphs_num, alllabels, maxh)

    allPaths = {}
    for gidx in range(graphs_num):
        adj_matrix = graph_datas[0][gidx]['am']
        nx_G = nx.from_numpy_matrix(adj_matrix)
        paths_graph = utils.path_pattern_generation(nx_G, depth)
        allPaths[gidx] = paths_graph

    PP = []
    for run in range(maxh):
        labels = alllabels[run]
        alllabeledpaths = []
        for gidx in range(graphs_num):
            paths = allPaths[gidx]
            label = labels[gidx]
            tmp_labeledpaths = utils.path_pattern_labeling(paths, label)
            alllabeledpaths.append(tmp_labeledpaths)
        dictionary = corpora.Dictionary(alllabeledpaths)
        # print(dictionary)

        # filtrated_graph_datas = [utils.filtration_by_edge_attribute(weighted_graph_data) for weighted_graph_data in weighted_graph_datas]
        corpus = []
        filtrated_num = []
        for gidx in range(graphs_num):
            # filtrated_graph_data = filtrated_graph_datas[gidx]
            filtrated_graph_data = utils.filtration_by_edge_attribute(weighted_graph_datas[gidx])
            filtrated_num.append(len(filtrated_graph_data))
            for i in range(len(filtrated_graph_data) - 1):
                node_idx = utils.get_node_idx(filtrated_graph_data[i])
                nx_G = utils.ig_to_nx(filtrated_graph_data[i])
                paths_filtrated_graph = utils.path_pattern_generation(nx_G, depth, node_idx=node_idx)
                label = labels[gidx]
                tmp_filtrated_labeledpaths = utils.path_pattern_labeling(paths_filtrated_graph, label)
                # print(gensim.matutils.corpus2csc([dictionary.doc2bow(tmp_filtrated_labeledpaths)]))
                corpus.append(dictionary.doc2bow(tmp_filtrated_labeledpaths))
            corpus.append(dictionary.doc2bow(alllabeledpaths[gidx]))
        M = gensim.matutils.corpus2csc(corpus)
        M = M.T.todense()
        # print(np.array(M).shape)
        PP.append(M)

    embeddings = np.asarray(np.concatenate(PP, axis=1))
    print(embeddings.shape)

    graph_embeddings = []
    index = 0
    num_features = len(embeddings[0])
    for gidx in range(graphs_num):
        # n = len(filtrated_graph_datas[gidx])
        n = filtrated_num[gidx]
        # if n == 1:
        #     print(gidx)
        feature_matrix = np.zeros((n, num_features))
        for num in range(n):
            feature_matrix[num, :] = embeddings[num + index, :]
        index += n
        graph_embeddings.append(feature_matrix)

    y = utils.get_graph_label(dataset)

    M1 = utils.compute_wasserstein_distance_matrix(graph_embeddings)

    time_end = time.time()

    print('kernel matrix(ds=%s, k=%d, d=%d) is computed in %.2fs' % (dataset, maxh - 1, depth - 1, (time_end - time_start)))

    gridsearch = True
    crossvalidation = True

    utils.run_classification(M1, y, gridsearch, crossvalidation)