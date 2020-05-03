import operator
import os
import random
import time
import networkx as nx
import pickle

import config


def generate_cascades(
        observation_time, prediction_time,
        filename, filename_filtered, filename_sorted_filtered,
        filename_train, filename_val, filename_test,
        filename_shortest_train, filename_shortest_val, filename_shortest_test,
        seed, shuffle_train=True):


    # open data file and start to write
    with open(filename) as file, \
            open(filename_filtered, 'w') as file_filtered:

        cascades_type = dict()  # 0 for train, 1 for val, 2 for test
        cascades_publish_time_dict = dict()
        cascade_total = 0
        cascade_valid_total = 0

        for line in file:
            # split the cascades into 5 parts
            cascade_total += 1
            parts = line.split('\t')
            cascade_id = parts[0]
            hour = int(time.strftime('%H', time.localtime(float(parts[2]))))
            n_retweet = int(parts[3])
            paths = parts[4].split(' ')

            # for Weibo dataset only, 18 for t_o of 0.5 hour and 19 for t_o of 1 hour
            if hour < 8 or hour >= 18:
                continue

            observation_path = list()
            p_o = 0
            for p in paths:
                nodes = p.split(':')[0].split('/')
                
                # observed participants
                time_now = int(p.split(":")[1])
                if time_now < observation_time:
                    p_o += 1
                observation_path.append((nodes, time_now))

            # filter cascades which number of participants is less than 10 
            # before observation time
            if p_o < 10:
                continue

            observation_path.sort(key=lambda tup: tup[1])

            # for each cascade, save their publish time
            cascades_publish_time_dict[cascade_id] = int(parts[2])

            o_path = list()

            for i in range(len(observation_path)):
                nodes = observation_path[i][0]
                t = observation_path[i][1]
                o_path.append('/'.join(nodes) + ':' + str(t))

            # write data into the targeted file, if they are not excluded

            line = parts[0] + '\t' + parts[1] + '\t' + parts[2] + '\t' \
                   + parts[3] + '\t' + ' '.join(o_path) + '\n'
            file_filtered.write(line)
            cascade_valid_total += 1

    with open(filename_filtered, 'r') as file_filtered, \
            open(filename_train, 'w') as file_train, \
            open(filename_val, 'w') as file_val, \
            open(filename_test, 'w') as file_test, \
            open(filename_shortest_train, 'w') as file_shortest_train, \
            open(filename_shortest_val, 'w') as file_shortest_val, \
            open(filename_shortest_test, 'w') as file_shortest_test:

        def sort_cascades_by_time():
            sorted_message_time = sorted(cascades_publish_time_dict.items(),
                                         key=operator.itemgetter(1))

            count = 0
            for (key, value) in sorted_message_time:
                if count < cascade_valid_total * .7:
                    cascades_type[key] = 0  # training set, 70%
                elif count < cascade_valid_total * .85:
                    cascades_type[key] = 1  # validation set, 15%
                else:
                    cascades_type[key] = 2  # test set, 15%
                count += 1

        # shuffle all dataset
        def shuffle_cascades():
            shuffled_message_time = list(cascades_publish_time_dict.keys())
            random.seed(seed)
            random.shuffle(shuffled_message_time)

            count = 0
            for key in shuffled_message_time:
                if count < cascade_valid_total * .7:
                    cascades_type[key] = 0  # training set, 70%
                elif count < cascade_valid_total * .85:
                    cascades_type[key] = 1  # validation set, 15%
                else:
                    cascades_type[key] = 2  # test set, 15%
                count += 1

        if shuffle_train:
            shuffle_cascades()
        else:
            sort_cascades_by_time()

        print("Number of valid cascades: {}/{}"
              .format(cascade_valid_total, cascade_total))

        filtered_data_train = list()
        filtered_data_val = list()
        filtered_data_test = list()
        for line in file_filtered:
            cascade_id = line.split('\t')[0]
            if cascades_type[cascade_id] == 0:
                filtered_data_train.append(line)
            elif cascades_type[cascade_id] == 1:
                filtered_data_val.append(line)
            elif cascades_type[cascade_id] == 2:
                filtered_data_test.append(line)
        
        print("Number of valid train cascades: {}".format(len(filtered_data_train)))
        print("Number of valid   val cascades: {}".format(len(filtered_data_val)))
        print("Number of valid  test cascades: {}".format(len(filtered_data_test)))

        if shuffle_train:
            random.seed(seed)
            random.shuffle(filtered_data_train)

        # write shuffled train data with sorted val & test data
        with open(filename_sorted_filtered, 'w') as file_sorted_filtered:
            for item in filtered_data_train:
                file_sorted_filtered.write(item)
            for item in filtered_data_val:
                file_sorted_filtered.write(item)
            for item in filtered_data_test:
                file_sorted_filtered.write(item)

        def file_shortest_write(file_name):
            """Write data into file_shortest_train/validation/test.txt. """
            file_name.write(cascade_id + '\t'
                            + '\t'.join(observation_path) + '\n')

        def file_write(file_name):
            """Write data into file_train/validation/test.txt. """
            file_name.write(cascade_id + '\t' + parts[1] + '\t' + parts[2]
                            + '\t' + str(len(observation_path)) + '\t'
                            + ' '.join(edges) + '\t' + ' '.join(labels) + '\n')

        with open(filename_sorted_filtered, 'r') as file_sorted_filtered:
            for line in file_sorted_filtered:
                # split the message into 5 parts as we just did
                parts = line.split('\t')
                cascade_id = parts[0]
                observation_path = list()
                labels = list()
                edges = set()
                paths = parts[4].split(' ')

                for i in range(len(prediction_time)):
                    labels.append(0)
                for p in paths:
                    nodes = p.split(':')[0].split('/')

                    time_now = int(p.split(":")[1])
                    if time_now < observation_time:
                        observation_path.append(",".join(nodes)
                                                + ":"
                                                + str(time_now))
                        # add edges' information
                        for i in range(1, len(nodes)):
                            edges.add(nodes[i - 1] + ":" + nodes[i] + ":1")
                    # add labels depends on prediction_time, e.g., 24 hours
                    for i in range(len(prediction_time)):
                        if time_now < prediction_time[i]:
                            labels[i] += 1

                # calculate the incremental prediction
                for i in range(len(labels)):
                    labels[i] = str(labels[i] - len(observation_path))

                # write files by cascade type
                # 0 to train, 1 to validate, 2 to test
                if cascade_id in cascades_type \
                        and cascades_type[cascade_id] == 0:
                    file_shortest_write(file_shortest_train)
                    file_write(file_train)
                    # file_label_write(file_label)
                elif cascade_id in cascades_type \
                        and cascades_type[cascade_id] == 1:
                    file_shortest_write(file_shortest_val)
                    file_write(file_val)
                elif cascade_id in cascades_type \
                        and cascades_type[cascade_id] == 2:
                    file_shortest_write(file_shortest_test)
                    file_write(file_test)

    g = nx.Graph()

    with open(filename, 'r') as f:
        for line in f:
            line.strip()
            parts = line.split('\t')
            paths = parts[4].strip().split(' ')
            for path in paths:
                nodes = path.split(':')[0].split('/')
                if len(nodes) < 2:
                    g.add_node(nodes[-1])
                else:
                    g.add_edge(nodes[-1], nodes[-2])

    print("Number of nodes in global graph:", g.number_of_nodes())
    print("Number of edges in global graph:", g.number_of_edges())

    with open(config.global_graph, 'wb') as f:
        pickle.dump(g, f)

    os.remove(filename_filtered)
    os.remove(filename_sorted_filtered)

    print('Finished')


if __name__ == "__main__":
    time_start = time.time()
    print('Start to processing CasFlow!\n')

    generate_cascades(config.observation_time,
                      config.prediction_time,
                      config.cascade_path,
                      config.cascade_filtered_path,
                      config.cascade_sorted_filtered_path,
                      config.cascade_train,
                      config.cascade_validation,
                      config.cascade_test,
                      config.cascade_shortestpath_train,
                      config.cascade_shortestpath_validation,
                      config.cascade_shortestpath_test, 
                      config.seed)

    time_end = time.time()
    print('Processing Time: {:.2f}s'.format(time_end - time_start))
