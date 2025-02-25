import numpy as np
from bertopic import BERTopic
import argparse
import guidedlda
import re
from gensim import corpora
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import corpus2dense
from nltk.corpus import stopwords
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer

from evaluation import *
from load_save_data import *

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-dataset', help='dataset to run topic modeling(clinc_domain/20newsgroups)',\
                        required=True, type=str)
    parser.add_argument('-scale', help='dataset scale', choices=['large', 'small'], default='small', type=str)
    parser.add_argument('-n_cluster', help='number of cluster', required=True, default=3, type=int)
    args = parser.parse_args()

    data = load_dataset(args.dataset, args.scale)
    num_cluster = args.n_cluster
    labels = [d['label'] for d in data]
    label_ids, _ = convert_label_to_ids(labels)
    manual_stop_list = []

    results = {'LLM_Refinement': {}, 
               'BERTopic': {}, 
               'LDA': {}, 
               'SeededLDA': {}, 
               'CorEX': {}}

    # LLM Refinement Metrics
    #################################################################
    print('\nProcessing LLM Refinement...')
    # phi
    topic_words_dict_list = load_ctfidf(args.dataset)
    
    llm_npmi = calculate_npmi(topic_words_dict_list[:num_cluster])
    llm_topic_diversity = calculate_topic_diversity(topic_words_dict_list[:num_cluster])

    # theta
    seeded_preds = load_clustering_result(args.dataset, num_cluster)
    llm_all_measures = {'ACC': [], 'NMI': []}
    
    seeded_preds = np.asarray(seeded_preds)
    measures = clustering_score(label_ids, seeded_preds)
    for k in measures:
        llm_all_measures[k].append(measures[k])

    results['LLM_Refinement']['NPMI'] = llm_npmi
    results['LLM_Refinement']['Topic_Diversity'] = llm_topic_diversity
    results['LLM_Refinement']['NMI'] = llm_all_measures['NMI'][0]
    results['LLM_Refinement']['Accuracy'] = llm_all_measures['ACC'][0]
    #################################################################

    # BERTopic Metrics
    #################################################################
    print('\nProcessing BERTopic...')
    # phi
    docs = []
    # Iterate through each dictionary and extract the input values
    for item in data:
        docs.append(item['input'])

    topic_model = BERTopic(top_n_words=20, nr_topics=num_cluster)
    topics, probs = topic_model.fit_transform(docs)

    topic_res = topic_model.get_topics()
    bertopic_output = [dict(words) for words in topic_res.values()]

    bertopic_npmi = calculate_npmi(bertopic_output)
    bertopic_topic_diversity = calculate_topic_diversity(bertopic_output)

    # theta
    bertopic_all_measures = {'ACC': [], 'NMI': []}
    topics = np.asarray(topics)

    bertopic_measures = clustering_score(label_ids, topics)
    for k in bertopic_measures:
        bertopic_all_measures[k].append(bertopic_measures[k])

    results['BERTopic']['NPMI'] = bertopic_npmi
    results['BERTopic']['Topic_Diversity'] = bertopic_topic_diversity
    results['BERTopic']['NMI'] = bertopic_all_measures['NMI'][0]
    results['BERTopic']['Accuracy'] = bertopic_all_measures['ACC'][0]
    #################################################################

    # LDA Metrics
    #################################################################
    print('\nProcessing LDA...')
    # phi
    docs = []
    # Iterate through each dictionary and extract the input values
    for item in data:
        docs.append(item['input'])

    # Remove all punctuation and numbers
    docs = [re.sub(r'\d+|[^\w\s]', '', doc) for doc in docs]

    # Preprocess
    texts = [[word for word in document.lower().split() if word not in stopwords.words('english') and word not in manual_stop_list]
            for document in docs]

    # Create a vocabulary list
    dictionary = corpora.Dictionary(texts)

    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_cluster)

    # Print topics
    topics = lda_model.print_topics(num_words=10)

    # Save each word and its score in a list of dict topics_dict
    topics_dict = [{dictionary[id]: round(freq, 3) for id, freq in lda_model.get_topic_terms(topicid)} for topicid in range(lda_model.num_topics)]

    # Clustering result
    topics_assignment = [max(lda_model[doc], key=lambda x: x[1])[0] for doc in corpus]
    topics_assignment = np.asarray(topics_assignment)

    LDA_npmi = calculate_npmi(topics_dict)
    LDA_topic_diversity = calculate_topic_diversity(topics_dict)

    # theta
    LDA_all_measures = {'ACC': [], 'NMI': []}

    LDA_measures = clustering_score(label_ids, topics_assignment)
    for k in LDA_measures:
        LDA_all_measures[k].append(LDA_measures[k])

    results['LDA']['NPMI'] = LDA_npmi
    results['LDA']['Topic_Diversity'] = LDA_topic_diversity
    results['LDA']['NMI'] = LDA_all_measures['NMI'][0]
    results['LDA']['Accuracy'] = LDA_all_measures['ACC'][0]
    #################################################################

    # SeededLDA Metrics
    #################################################################
    print('\nProcessing SeededLDA...')
    # phi
    docs = []
    # Iterate through each dictionary and extract the input values
    for item in data:
        docs.append(item['input'])

    # Preprocess
    docs = [re.sub(r'\d+|[^\w\s]', '', doc) for doc in docs]
    docs = [' '.join([word for word in doc.lower().split() if word not in stopwords.words('english') and word not in manual_stop_list]) for doc in docs]

    sample_docs2 =  [doc.split() for doc in docs]
    dictionary = Dictionary(sample_docs2)
    corpus = [dictionary.doc2bow(doc) for doc in sample_docs2]
    vocab = tuple(dictionary.token2id.keys())
    word2id = dictionary.token2id

    # Load seed topics
    seed_topic_list = load_seedwords(args.dataset)
    seed_topics = {}
    for t_id, word in enumerate(seed_topic_list):
        seed_topics[word2id[word]] = t_id

    # Since GuidedLDA requires the DTM format as input, using the `corpus2dense()` method to convert the corpus into the required format
    X = corpus2dense(corpus, len(dictionary), len(corpus)).T.astype(np.int64)

    model = guidedlda.GuidedLDA(n_topics=num_cluster, n_iter=100, random_state=7, refresh=20)
    model.fit(X, seed_topics=seed_topics, seed_confidence=1)

    # Organize/Display the topic model results
    n_top_words = 10
    topic_word = model.topic_word_

    # List of dict of each word and its score
    topics_dict = [{vocab[id]: freq for id, freq in enumerate(topic_dist)} for topic_dist in topic_word]

    # List of clustering result
    topics_assignment = model.transform(X).argmax(axis=1)
    topics_assignment = np.asarray(topics_assignment)

    seededLDA_npmi = calculate_npmi(topics_dict)
    seededLDA_topic_diversity = calculate_topic_diversity(topics_dict)

    # theta
    seededLDA_all_measures = {'ACC': [], 'NMI': []}

    seededLDA_measures = clustering_score(label_ids, topics_assignment)
    for k in seededLDA_measures:
        seededLDA_all_measures[k].append(seededLDA_measures[k])

    results['SeededLDA']['NPMI'] = seededLDA_npmi
    results['SeededLDA']['Topic_Diversity'] = seededLDA_topic_diversity
    results['SeededLDA']['NMI'] = seededLDA_all_measures['NMI'][0]
    results['SeededLDA']['Accuracy'] = seededLDA_all_measures['ACC'][0]
    #################################################################

    # Anchored CorEX Metrics
    #################################################################
    print('\nProcessing Anchored CorEX...')
    # phi
    docs = []
    # Iterate through each dictionary and extract the input values
    for item in data:
        docs.append(item['input'])

    # Remove punctuations, numbers, and stop words 
    docs = [re.sub(r'\d+|[^\w\s]', '', doc) for doc in docs]
    docs = [' '.join([word for word in doc.lower().split() if word not in stopwords.words('english') and word not in manual_stop_list]) for doc in docs]

    # Ctreat word freq matric
    vectorizer = CountVectorizer(binary=True)
    doc_word = vectorizer.fit_transform(docs)
    words = list(np.asarray(vectorizer.get_feature_names_out()))

    # Initialize CorEx model
    topic_model = ct.Corex(n_hidden=num_cluster, words=words, seed=1)
    topic_model.fit(doc_word, words=words, anchors=None, docs=docs)

    # Get the list of dict of each word and its score
    topics = topic_model.get_topics()
    topic_words_scores = [{word: mi for word, mi, _ in topic} for topic in topics]

    # Obtain the clustering results as a list and convert the 2D boolean array into a 1D array
    labels_single = np.argmax(topic_model.labels, axis=1)

    corEX_npmi = calculate_npmi(topic_words_scores)
    corEX_topic_diversity = calculate_topic_diversity(topic_words_scores)

    # theta
    corEX_all_measures = {'ACC': [], 'NMI': []}

    corEX_measures = clustering_score(label_ids, labels_single)
    for k in corEX_measures:
        corEX_all_measures[k].append(corEX_measures[k])

    results['CorEX']['NPMI'] = corEX_npmi
    results['CorEX']['Topic_Diversity'] = corEX_topic_diversity
    results['CorEX']['NMI'] = corEX_all_measures['NMI'][0]
    results['CorEX']['Accuracy'] = corEX_all_measures['ACC'][0]
    #################################################################

    print('\n---------- Results ----------\n', results)
    save_performance(args.dataset, num_cluster, results)