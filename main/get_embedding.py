import logging
from sentence_transformers import SentenceTransformer

from load_save_data import *

logging.basicConfig(level=logging.INFO)

def get_embedding(args):
    print("--GET EMBEDDING START--")
    data = load_dataset(args.dataset, args.scale)

    texts = []
    for datum in data:
        texts.append(datum['input'])

    # Get doc embedding
    model = SentenceTransformer('BAAI/bge-m3')
    print('Starting bge-m3')
    print('total instance number: ', len(texts))
    embeds = model.encode(sentences=texts, show_progress_bar=True, normalize_embeddings=True, batch_size=1)

    # Seed words list
    seed_words = load_seedwords(args.dataset)

    # Get seed word embeddings
    embeds_seed = model.encode(sentences=seed_words, show_progress_bar=True, normalize_embeddings=True)

    # Save data
    save_doc_embedding(embeds, args.dataset, args.scale)
    save_seeds_embedding(embeds_seed, args.dataset, len(seed_words))

    print("--GET EMBEDDING DONE--")

    return len(seed_words)