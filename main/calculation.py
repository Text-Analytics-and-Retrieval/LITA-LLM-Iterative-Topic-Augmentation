import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.corpus import stopwords
import re

def entropy(vals):
    vals = np.asarray(vals)
    vals /= vals.sum()
    return - (vals * np.log(vals)).sum()

# c-TF-IDF
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    sorted_items = sorted_items[:topn]
    score_vals = []
    feature_vals = []
    
    for idx, score in sorted_items:
        fname = feature_names[idx]
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results

def calculate_ctfidf(dataset, cluster_results, documents):
    # Remove punctuation, numbers, and stop words
    if dataset == '20newsgroups':
        manual_stop_list = ['edu', 'lines', 'subject', 'from', 'organization', 're', 'x', 'would', 'could', 'one', 'article', 'dont', 'think', 'like', 'writes', 'nntppostinghost', 'b', 'v']
    else:
        manual_stop_list = []
    documents = [re.sub(r'\d+|[^\w\s]', '', doc) for doc in documents]
    stop_words = set(stopwords.words('english'))
    documents = [' '.join([word for word in doc.lower().split() if word not in stop_words and word not in manual_stop_list]) for doc in documents]

    # Calculate word freq by CountVectorizer
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(documents)
    # Before calculating c-TF-IDF，get all feature names
    feature_names = cv.get_feature_names_out()

    # Calculate TF-IDF by TfidfTransformer
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    tfidf_transformer.fit(word_count_vector)

    c_tf_idf = []
    for i, cluster in enumerate(cluster_results):
        tf_idf_vector = tfidf_transformer.transform(cv.transform([' '.join(cluster)]))
        sorted_items = sort_coo(tf_idf_vector.tocoo())
        keywords = extract_topn_from_vector(feature_names, sorted_items, 20)
        c_tf_idf.append(keywords)

    # Load stop words
    nltk.download('stopwords')
    stops = stopwords.words('english')

    topic_words_dict_list = []
    # Find topic words of each cluster
    for i, keywords in enumerate(c_tf_idf):
        topic_words_dict_list.append(keywords)

    # Filter stop words
    filtered_topic_words_dict_list = [{k: v for k, v in d.items() if k not in stops} for d in topic_words_dict_list]

    # Print topic words of each cluster
    for i, keywords in enumerate(filtered_topic_words_dict_list):
        print(f'Cluster {i+1} topic words: {", ".join(keywords)}')

    return filtered_topic_words_dict_list

# closest clusters
def get_closest_clusters(seeded_n_clusters, seeded_cluster_centers, X, args):
    seeded_num_closest = max(5, round(seeded_n_clusters * args['close_cluster_prop']))
    seeded_options = []
    seeded_entropies = []
    for idx in range(len(X)):
        # Compute the Euclidean distance from the current data point to all cluster centers
        dist = ((X[idx] - seeded_cluster_centers) ** 2).sum(-1)
        # Convert the distances into a probability distribution using the softmax method
        prob = (1 + dist) ** (-1)
        prob /= prob.sum()
        # Rank all clusters based on probability and select the top seeded_num_closest clusters with the highest probabilities
        sorted_prob = np.argsort(prob)[::-1][:seeded_num_closest]
        seeded_options.append(sorted_prob) # most probable cluster index
        seeded_entropies.append(entropy(prob[sorted_prob])) # entropy with most probable clusters
    
    # The certain proportion of the top instances with the highest entropy are selected and stored in seeded_sorted_ent
    if args['filter_first_prop'] > 0:
        seeded_sorted_ent = np.argsort(seeded_entropies)[::-1][int(len(X) * args['filter_first_prop']):int(len(X) * args['large_ent_prop'])]
    else:
        seeded_sorted_ent = np.argsort(seeded_entropies)[::-1][:int(len(X) * args['large_ent_prop'])]
    if args['shuffle_inds']:
        np.random.shuffle(seeded_sorted_ent)
    
    return seeded_sorted_ent

# Calculate distance to pick ambiguous instances
def select_ambiguous_indices(seeded_cluster_centers, X, threshold):
    # Compute the Euclidean distance from each data point to all cluster centers
    distances = np.sqrt(((X[:, np.newaxis] - seeded_cluster_centers) ** 2).sum(axis=2))

    # Find the distance from each data point to its nearest and second-nearest cluster centers
    closest_clusters = np.partition(distances, kth=[0, 1], axis=1)[:, :2]

    # Compute the distance difference between the nearest and second-nearest cluster centers
    distance_differences = closest_clusters[:, 1] - closest_clusters[:, 0]

    # Determine how many data points are considered ambiguous instances based on a threshold
    ambiguous_indices = np.where(distance_differences <= threshold)[0]  # clinc_domain=0.15, news=0.07~0.1(c=23)
    return ambiguous_indices