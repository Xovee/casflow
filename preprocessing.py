import pickle
import time

import config
from utils.graphwave.graphwave import *
from utils.sparse_matrix_factorization import *


def sequence2list(filename):
    graphs = dict()
    with open(filename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')[:config.max_sequence+1]
            # put message/cascade id into graphs dictionary, value is a list
            graphs[walks[0]] = list()
            for i in range(1, len(walks)):
                nodes = walks[i].split(":")[0]
                time = walks[i].split(":")[1]
                graphs[walks[0]] \
                    .append([[int(x) for x in nodes.split(",")],
                             int(time)])

    return graphs


def read_labels_and_sizes(filename):
    labels = dict()
    sizes = dict()
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split('\t')
            # parts[-1] means the incremental popularity
            labels[parts[0]] = parts[-1]
            # parts[3] means the observed popularity
            sizes[parts[0]] = int(parts[3])
    return labels, sizes


def write_cascade(graphs, labels, sizes, length, filename, gg_emb,
                  weight=True):

    y_data = list()
    size_data = list()
    time_data = list()
    rnn_index = list()
    embedding = list()
    n_cascades = 0
    new_input = list()
    global_input = list()
    for key, graph in graphs.items():
        label = labels[key].split()
        y = int(label[0])
        temp_time = list()
        temp_index = list()
        temp_size = len(graph)
        for walk in graph:
            # save publish time into temp_time list
            temp_time.append(walk[1])
            # save length of walk into temp_index
            temp_index.append(len(walk[0]))
        y_data.append(y)
        size_data.append(temp_size)
        time_data.append(temp_time)
        rnn_index.append(temp_index)
        n_cascades += 1

    # padding the embedding
    embedding_size = config.gc_emd_size

    cascade_i = 0
    cascade_size = len(graphs)
    total_time = 0

    for key, graph in graphs.items():
        start_time = time.time()
        new_temp = list()
        global_temp = list()
        dg = nx.DiGraph()
        nodes_index = list()
        list_edge = list()
        cascade_embedding = list()
        global_embedding = list()
        times = list()
        t_o = config.observation_time
        for path in graph:
            t = path[1]
            if t >= t_o:
                continue
            nodes = path[0]
            if len(nodes) == 1:
                nodes_index.extend(nodes)
                times.append(1)
                continue
            else:
                nodes_index.extend([nodes[-1]])
            if weight:
                edge = (nodes[-1], nodes[-2], (1 - t / t_o))
                times.append(1 - t / t_o)
            else:
                edge = (nodes[-1], nodes[-2])
            list_edge.append(edge)
        if weight:
            dg.add_weighted_edges_from(list_edge)
        else:
            dg.add_edges_from(list_edge)
        nodes_index_unique = list(set(nodes_index))
        nodes_index_unique.sort(key=nodes_index.index)
        g = dg

        d = embedding_size / (2 * config.number_of_s)
        if embedding_size % 4 != 0:
            raise ValueError
        chi, _, _ = graphwave_alg(g, np.linspace(0, 100, int(d)),
                                  taus='auto', verbose=False,
                                  nodes_index=nodes_index_unique,
                                  nb_filters=config.number_of_s)
        for node in nodes_index:
            cascade_embedding.append(chi[nodes_index_unique.index(node)])
            global_embedding.append(gg_emb[id2row[node]])
        if weight:
            cascade_embedding = np.concatenate([np.reshape(times, (-1, 1)), np.array(cascade_embedding)[:, 1:]], axis=1)
        new_temp.extend(cascade_embedding)
        global_temp.extend(global_embedding)
        new_input.append(new_temp)
        global_input.append(global_temp)

        total_time += time.time() - start_time
        cascade_i += 1
        if cascade_i % 100 == 0:
            speed = total_time / cascade_i
            eta = (cascade_size - cascade_i) * speed
            print("{}/{}, eta: {:.2f} minutes".format(
                cascade_i, cascade_size, eta / 60))

    with open(filename, 'wb') as fin:
        pickle.dump((new_input, global_input, y_data), fin)


def get_max_size(sizes):
    max_size = 0
    for cascade_id in sizes:
        max_size = max(max_size, sizes[cascade_id])
    return max_size


def get_max_length(graphs):
    """ Get the max length among sequences. """
    max_length = 0
    for cascade_id in graphs:
        # traverse the graphs for max length sequence
        for sequence in graphs[cascade_id]:
            max_length = max(max_length, len(sequence[0]))
    return max_length


if __name__ == "__main__":
    
    time_start = time.time()

    # get the information of nodes/users of cascades
    graphs_train = sequence2list(config.cascade_shortestpath_train)
    graphs_val = sequence2list(config.cascade_shortestpath_validation)
    graphs_test = sequence2list(config.cascade_shortestpath_test)

    # get the information of labels and sizes of cascades
    labels_train, sizes_train = read_labels_and_sizes(config.cascade_train)
    labels_val, sizes_val = read_labels_and_sizes(config.cascade_validation)
    labels_test, sizes_test = read_labels_and_sizes(config.cascade_test)

    # find the max length of sequences
    len_sequence = max(get_max_length(graphs_train),
                       get_max_length(graphs_val),
                       get_max_length(graphs_test))
    print("Max length of sequence:", len_sequence)

    print("Cascade graph embedding size:", config.gc_emd_size)
    print("Number of scale s:", config.number_of_s)

    # load global graph and generate id2row
    with open(config.global_graph, 'rb') as f:
        gg = pickle.load(f)

    # sparse matrix factorization
    model = SparseMatrixFactorization(gg, config.gg_emd_size)
    features_matrix = model.pre_factorization(model.matrix, model.matrix)
    np.save(config.global_embedding, features_matrix)

    ids = [int(xovee) for xovee in gg.nodes()]

    id2row = dict()
    i = 0
    for id in ids:
        id2row[id] = i
        i += 1

    # load global graph embeddings
    gg_emb = np.load(config.global_embedding + '.npy')

    print("Start writing train set into file.")
    write_cascade(graphs_train, labels_train, sizes_train, len_sequence,
                    config.train, gg_emb=gg_emb)
    print("Start writing validation set into file.")
    write_cascade(graphs_val, labels_val, sizes_val,
                    len_sequence, 
                    config.val, gg_emb=gg_emb)
    print("Start writing test set into file.")
    write_cascade(graphs_test, labels_test, sizes_test, len_sequence,
                    config.test, gg_emb=gg_emb)

    time_end = time.time()
    print("Processing time: {0:.2f}s".format(time_end - time_start))
