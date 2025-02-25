import numpy as np
from collections import Counter
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

#  NPMI and Topic Diversity (phi)
#################################################################
def calculate_npmi(topic_words_dict_list):
    # Word freq
    word_frequency = Counter(word for topic_words_dict in topic_words_dict_list for word in topic_words_dict)
    total_frequency = sum(word_frequency.values())
    # Topic freq
    topic_frequency = [sum(topic_words_dict.values()) for topic_words_dict in topic_words_dict_list]
    # NPMI
    npmi_values = []
    for topic_words_dict, topic_freq in zip(topic_words_dict_list, topic_frequency):
        for word, freq in topic_words_dict.items():
            joint_prob = freq / total_frequency
            npmi = np.log(joint_prob / (word_frequency[word] / total_frequency * topic_freq / total_frequency)) / -np.log(joint_prob)
            npmi_values.append(npmi)
    # Avg. NPMI
    avg_npmi = np.mean(npmi_values)
    return avg_npmi

def calculate_topic_diversity(topic_words_dict_list, topk=25):
    # Obtain the topk words for each topic
    topk_words_per_topic = [sorted(topic_words_dict, key=topic_words_dict.get, reverse=True)[:topk] for topic_words_dict in topic_words_dict_list]
    # Topic diversity
    topic_diversity = len(set(word for topic_words in topk_words_per_topic for word in topic_words)) / (topk * len(topic_words_dict_list))
    return topic_diversity
#################################################################

# Clustering Evaluation (theta)
#################################################################
def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w

def clustering_accuracy_score(y_true, y_pred):
    ind, w = hungray_aligment(y_true, y_pred)
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc

def clustering_score(y_true, y_pred):
    return {'ACC': clustering_accuracy_score(y_true, y_pred)*100,
            'NMI': normalized_mutual_info_score(y_true, y_pred)*100}

def convert_label_to_ids(labels):
    unique_labels = list(set(labels))
    n_clusters = len(unique_labels)
    label_map = {l: i for i, l in enumerate(unique_labels)}
    label_ids = [label_map[l] for l in labels]
    return np.asarray(label_ids), n_clusters
#################################################################