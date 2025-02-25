import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from load_save_data import *
from calculation import *
from tools import *

def remove_no_move(topics, clusters_result, amb_pred_df):
    # Swap the keys and values of the `topics` dictionary, as we need to look up the corresponding key based on the value of the content
    inverse_topics = {v: k for k, v in topics.items()}

    # Set target_cluster by content value
    amb_pred_df['target_cluster'] = amb_pred_df['content'].map(inverse_topics)

    # Return the corresponding clusters_result value based on the query_idx
    def get_cluster_result(query_idx):
        return clusters_result[query_idx]

    # Use .apply() to add 'original_c_idx' column
    amb_pred_df['original_c_idx'] = amb_pred_df['query_idx'].apply(get_cluster_result)

    # Add column 'original_topic', the value is based on the 'query_idx' corresponding to the values of the topics
    amb_pred_df['original_topic'] = amb_pred_df['original_c_idx'].astype(str).map(topics)

    # Remove row which has the same value of column 'content' and column 'original_topic'
    amb_pred_df = amb_pred_df.query("content != original_topic")

    return amb_pred_df

def homeless_agglo_clustering(X, amb_pred_df, args):
    # Get homeless ambiguous instances embeddings
    homeless_df = amb_pred_df[amb_pred_df['content'] == 'None']
    homeless_idx = homeless_df['query_idx'].tolist()
    homeless_embeds = [X[i] for i in range(len(X)) if i in homeless_idx]
    homeless_embeds = np.asarray(homeless_embeds)

    if len(homeless_embeds) < 2:
        return homeless_df, [], homeless_embeds

    # Agglomerative Clustering
    homeless_clustering = AgglomerativeClustering(distance_threshold=args.agglo_distance_threshold, n_clusters=None)
    homeless_labels = homeless_clustering.fit_predict(homeless_embeds)
    return homeless_df, homeless_labels, homeless_embeds

def get_topic_words(homeless_df, homeless_labels, data, args):
    # Find text of new clusters
    homeless_df['new_cluster'] = homeless_labels
    grouped_idx_lists = [group['query_idx'].tolist() for name, group in homeless_df.groupby('new_cluster')]

    # Create a list to save this text of new clusters
    homeless_docs_list = []

    for group in grouped_idx_lists:
        temp_list = []
        for doc_idx in group:
            temp_list.append(data[doc_idx]['input'])
        homeless_docs_list.append(temp_list)
    print('Number of new cluster: ', len(homeless_docs_list))
    print()

    # Merge sentences of each cluster into a docment
    documents = [' '.join(cluster) for cluster in homeless_docs_list]

    # Calculate c-TF-IDF, find topic words
    filtered_topic_words_dict_list = calculate_ctfidf(args.dataset, homeless_docs_list, documents)
    
    return filtered_topic_words_dict_list
  
def generate_topic_name(topic_word_lists, model_name, openai_api_key, openai_org):
    topic_name_list = []
    new_cluster_topics = []
    for word_list in topic_word_lists:
        # Use join() to transfer list into str
        word_list_str = ', '.join(word_list)

        generate_topic_name_prompt = f"""
Please generate a general topic name that can best describe the following word list, and the identified topic name should be less than two words\n\n.
Word list: [{word_list_str}]\n\n
Note: If the generated topic name is 'None', please casually select a word in the word list.
"""

        messages = [
            {"role": "user", "content": generate_topic_name_prompt}
        ]
        completion, error = delayed_completion(openai_api_key, openai_org, delay_in_seconds=1, max_trials=3, model=model_name, messages=messages, max_tokens=10, temperature=1)
        if completion is None:
            print(error)
            continue

        content = completion.choices[0].message.content
        topic_name_list.append(content)
    
    for i in range(0, len(topic_name_list)):
        new_cluster_topics.append({i: topic_name_list[i]})
    return topic_name_list, new_cluster_topics

def update_topics(topic_name_list, args):
    # Load current topics.json（seed topics）
    curr_topics = load_topics(args.dataset)
    old_topic_dict = load_topics(args.dataset)

    old_topic_names = list(old_topic_dict.values())

    # Find the max key currentlly
    max_key = max(int(k) for k in curr_topics.keys())

    # Add every element in topic_name_list into curr_topics
    for topic in topic_name_list:
        max_key += 1
        curr_topics[str(max_key)] = topic

    # Save dict into .json to update topics.json
    save_topics(args.dataset, curr_topics)

def update_seed_embedding(homeless_labels, homeless_embeds, new_cluster_topics):
    # Calculate the avg embedding of each cluster as new seed embeddings
    embed_cluster = {}  # {0: [arrays], 1: [arrays], ...}
    avg_embed_cluster = {}  # {0: avg_embed array, 1: avg_embed array, ...}
    for cluster_num in range(0, homeless_labels.max()+1):
        embed_cluster[cluster_num] =  []
        for idx, cluster in enumerate(homeless_labels):
            if cluster == cluster_num:
                embed_cluster[cluster_num].append(homeless_embeds[idx])
        # Only new topics that are self-contained can be added to the seed embeddings
        if cluster_num in [key for dict in new_cluster_topics for key in dict.keys()]:
            # Calculate avg embedding of new clusters
            avg_embed_cluster[cluster_num] = np.mean(embed_cluster[cluster_num], axis=0)

    return embed_cluster, avg_embed_cluster

def reaverage_seed_embedding(X, amb_pred_df, clusters_result, avg_embed_cluster):
    # Show all embeddings of clusters that have new docments
    newhome_df = amb_pred_df[amb_pred_df['content'] != 'None']
    newhome_df = newhome_df.dropna(subset=['target_cluster'])

    old_clusters = list(clusters_result)
    old_embed_cluster = {}  # {0: [arrays], 1: [arrays], ...}
    for idx, cluster_num in enumerate(old_clusters):
        if cluster_num not in list(old_embed_cluster.keys()):
            old_embed_cluster[cluster_num] = []
        old_embed_cluster[cluster_num].append(X[idx])

    # Add the embeddings of ambiguous instances with new homes to the `old_embed_cluster` (including all embeddings of that cluster)
    def add_new_embed_into_old_cluster(row):
        key = int(row['target_cluster'])
        value = row['query_idx']
        old_embed_cluster[key] = np.vstack((old_embed_cluster[key], X[value]))

    newhome_df.apply(add_new_embed_into_old_cluster, axis=1)

    # Calculate the avg embedding of each cluster as new seed embeddings
    for key in old_embed_cluster.keys():
        # Transfer np.array in the list into 2D np.array
        arrays = np.array(old_embed_cluster[key])
        # Calculate avg
        mean_array = np.mean(arrays, axis=0)
        # Update dict
        old_embed_cluster[key] = mean_array
    
    # Add new seed embeddings into old seed embeddings
    seeds_embeds = np.vstack(list(old_embed_cluster.values()))  # old seed embeddings
    print('\nBefore refinement, number of seed embeddings: ', seeds_embeds.shape[0])
    for element in avg_embed_cluster.values():
        seeds_embeds = np.vstack((seeds_embeds, element))
    print('After refinement, number of seed embeddings: ', seeds_embeds.shape[0])
    return seeds_embeds


def refine(args):
    # Load all embedding
    X = load_embedding_from_hdf5(args.dataset, args.scale)
    # Load original seed embeddings
    seeds_embeds = load_embedding_from_npy(args.dataset, args.n_cluster)
    # Load clustering result last round
    clusters_result = load_clustering_result(args.dataset, args.n_cluster)
    # Load current seed topics
    topics = load_topics(args.dataset)
    # Load dataset
    data = load_dataset(args.dataset, args.scale)
    # Load predict results of ambiguous instances last round
    amb_pred = load_last_round_ambs(args)
    # ambiguous instances
    amb_pred_df = pd.DataFrame(amb_pred)[['query_idx', 'content']]

    # Remove docments that do not move to the other cluster
    amb_pred_df = remove_no_move(topics, clusters_result, amb_pred_df)

    # Cluster the homelesses by agglomerative clustering
    homeless_df, homeless_labels, homeless_embeds = homeless_agglo_clustering(X, amb_pred_df, args)
    if len(homeless_labels) == 0:
        return 0, len(topics)
    
    # Find topic words of new clusters
    filtered_topic_words_dict_list = get_topic_words(homeless_df, homeless_labels, data, args)
    
    # Obtain topic word lists for LLM to generate topic names
    topic_word_lists = []
    for word_dict in filtered_topic_words_dict_list:
        words = list(word_dict.keys())
        topic_word_lists.append(words)

    # Generate topic names
    print('\nGenerating topic names...')
    topic_name_list, new_cluster_topics = generate_topic_name(topic_word_lists, args.model_name, args.openai_api_key, args.openai_org)
    if topic_name_list == ['None']:
        return 0, len(topics)
    print('\nNew topic names: ', topic_name_list)

    # Add the newly generated topic name into `topics.json`.
    update_topics(topic_name_list, args)
    
    # Obtain the average embedding of those homeless clusters that will not be merged, and add them to the seed word embeddings (i.e., `seeds_embeds_c=?.npy`)
    print('\nUpdating seed embeddings...')
    embed_cluster, avg_embed_cluster = update_seed_embedding(homeless_labels, homeless_embeds, new_cluster_topics)

    # For those instances that have found a new home, recalculate the average of the seed embeddings
    seeds_embeds = reaverage_seed_embedding(X, amb_pred_df, clusters_result, avg_embed_cluster)
    refined_n_cluster = seeds_embeds.shape[0]
    save_seeds_embedding(seeds_embeds, args.dataset, refined_n_cluster)

    return len(new_cluster_topics), refined_n_cluster