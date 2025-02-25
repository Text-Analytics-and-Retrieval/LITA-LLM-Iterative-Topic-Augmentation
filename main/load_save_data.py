import json
import h5py
import numpy as np
import os

def load_dataset(dataset, scale):
    with open(f'../datasets/{dataset}/{scale}.jsonl', 'r') as f:
        data = [json.loads(l) for l in f]
    return data

def load_embedding_from_hdf5(dataset, scale='small'):
    with h5py.File(f'../datasets/{dataset}/{scale}_embeds.hdf5', 'r') as f:
        X = f['embeds']
        X = np.asarray(X)
    return X

def load_embedding_from_npy(dataset, num_cluster):
    embeds_seed = np.load(f'../results/seeds_embeddings/{dataset}/seeds_embeds_c={num_cluster}.npy')
    return embeds_seed

def load_seedwords(dataset):
    with open(f'utils/seed_words.json', 'r', encoding='utf-8') as file:
        seed_words = json.load(file)
    return seed_words[dataset]

def load_topics(dataset):
    with open(f'../results/topics/{dataset}/topics.json', 'r', encoding='utf-8') as file:
        topics = json.load(file)
    return topics

def load_predict_data(dataset, scale, model_name, n_cluster):
    data_path = f'../results/ambiguous_selecting_results/{dataset}_s=small.json'
    pred_path = f"seeded_{dataset}_s={scale}-{model_name}-pred_c={n_cluster}.json"
    pred_path = os.path.join(f"../results/predicted_topic_results/{dataset}", pred_path)
    if os.path.exists(pred_path):
        with open(pred_path, 'r') as f:
            data = json.load(f)
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
    return data, pred_path

def load_prompts(dataset):
    with open("utils/prompts.json", 'r') as f:
        prompts = json.load(f)
        task_prompt = prompts[dataset]
    return task_prompt

def load_last_round_ambs(args):
    with open(f"../results/predicted_topic_results/{args.dataset}/seeded_{args.dataset}_s={args.scale}-{args.model_name}-pred_c={args.n_cluster}.json", 'r') as f:
        data = json.load(f)
    return data

def load_clustering_result(dataset, topic_num):
    clusters_result = np.load(f'../results/clustering/{dataset}/clustering_result_c={topic_num}.npy')
    return clusters_result

def load_ctfidf(dataset):
    with open(f"../results/topics/{dataset}/ctfidf.json", "r", encoding="utf-8") as f:
        topic_words_dict_list = json.load(f)
    return topic_words_dict_list

def save_doc_embedding(embeds, dataset, scale='small'):
    with h5py.File(f'../datasets/{dataset}/{scale}_embeds.hdf5', 'w') as f:
        dset = f.create_dataset("embeds", data=embeds)

def save_seeds_embedding(embeds, dataset, topic_num):
    np.save(f'../results/seeds_embeddings/{dataset}/seeds_embeds_c={topic_num}.npy', embeds)

def save_clustering_result(embeds, dataset, topic_num):
    np.save(f'../results/clustering/{dataset}/clustering_result_c={topic_num}.npy', embeds)

def save_ambiguous_selecting_result(seeded_result, dataset, scale='small'):
    with open(f"../results/ambiguous_selecting_results/{dataset}_s={scale}.json", 'w') as f:
        json.dump(seeded_result, f)  

def save_predict_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def save_topics(dataset, curr_topics):
    with open(f'../results/topics/{dataset}/topics.json', 'w', encoding='utf-8') as f:
        json.dump(curr_topics, f, ensure_ascii=False, indent=4)

def save_ctfidf(dataset, dict_list):
    with open(f"../results/topics/{dataset}/ctfidf.json", "w", encoding="utf-8") as f:
        json.dump(dict_list, f, ensure_ascii=False, indent=4)

def save_performance(dataset, num_cluster, results):
    with open(f"../results/performance/{dataset}/results_c={num_cluster}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)