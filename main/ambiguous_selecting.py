import os
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

from load_save_data import *
from calculation import *
from evaluation import *

def select_amb(args):
    os.makedirs('results/ambiguous_selecting_results', exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.dataset, args.scale)

    inp = [d['input'] for d in data]

    X = load_embedding_from_hdf5(args.dataset, args.scale)
    X = StandardScaler().fit_transform(X)

    # Load seed words embeddings
    num_cluster = args.n_cluster
    embeds_seed = load_embedding_from_npy(args.dataset, num_cluster)

    seeded_clustering = MiniBatchKMeans(n_clusters=len(embeds_seed), init=embeds_seed, n_init=1).fit(X)

    # Check cluster sentences and topic words
    #################################################################
    seeded_preds = seeded_clustering.labels_

    seeded_n_clusters = len(set(seeded_preds))
    print("Number of clusters: %d" % seeded_n_clusters)

    # Use numpy save() to save clustering results to local file
    save_clustering_result(seeded_preds, args.dataset, num_cluster)

    seeded_cluster_centers = []
    seeded_class_member_inds = {}   # clustering results, including how many idx of instances each cluster has
    for i in range(seeded_n_clusters):
        seeded_class_member_mask = seeded_preds == i
        seeded_cluster_centers.append(X[seeded_class_member_mask].mean(0))
        seeded_class_member_inds[i] = np.where(seeded_class_member_mask)[0]
    seeded_cluster_centers = np.stack(seeded_cluster_centers)
    #################################################################

    # Calculate c-TF-IDF, find topic words
    #################################################################
    inds = seeded_class_member_inds
    cluster_results = []
    for key, value in inds.items():
        target_idx = value
        tmp_cluster_reslut = []
        for i, instance_idx in enumerate(inp):
            if i in target_idx:
                tmp_cluster_reslut.append(inp[i])
        cluster_results.append(tmp_cluster_reslut)
    
    # Merge the sentences of each cluster into a single document
    documents = [' '.join(cluster) for cluster in cluster_results]

    # c-TF-IDF
    filtered_topic_words_dict_list = calculate_ctfidf(args.dataset, cluster_results, documents)
    #################################################################

    # Store the c-TF-IDF of words for performance calculation
    save_ctfidf(args.dataset, filtered_topic_words_dict_list)
    
    # Load all current topic names
    topics = load_topics(args.dataset)

    # Convert value of dict into list
    topics = list(topics.values())

    topic_names_and_words = {}  # {"topic_name1": [word1, word2, ...], "topic_name2": [word1, word2, ...], ...}
    for idx, topic_name in enumerate(topics):
        topic_names_and_words[topic_name] = list(filtered_topic_words_dict_list[idx].keys())

    # Ambiguous instances selecting
    selecting_method = 'distance'
    if selecting_method == 'entropy':
        ambiguous_indices = get_closest_clusters(seeded_n_clusters, seeded_cluster_centers, X, args)
    else:
        ambiguous_indices = select_ambiguous_indices(seeded_cluster_centers, X, args.ambiguous_threshold)

    # Sum up ambiguous instances selecting results
    seeded_result = []
    for query_idx in ambiguous_indices:
        input_txt = "Query: " + inp[query_idx] + "\nChoice"
        seeded_result.append({
            "input": input_txt,
            "options": topic_names_and_words,
            "task": args.dataset,
            "query_idx": int(query_idx)
        })

    print("Total number of ambiguous instance: ", len(seeded_result))   
    save_ambiguous_selecting_result(seeded_result, args.dataset, args.scale)
