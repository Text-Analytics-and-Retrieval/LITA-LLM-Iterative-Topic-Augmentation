import argparse
from get_embedding import get_embedding
from ambiguous_selecting import select_amb
from predict_topic import predict
from refinement import refine

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('-model_name', help='LLM model name', default='gpt-3.5-turbo',type=str)
    parser.add_argument('-dataset', help='dataset to run topic modeling(clinc_domain/20newsgroups)',\
                        required=True, type=str)
    parser.add_argument('-ambiguous_threshold', help='threshold to select ambiguous instances', default=0.05, type=float)
    parser.add_argument('-seed', help='random seed', default=100, type=int)
    parser.add_argument('-scale', help='dataset scale', choices=['large', 'small'], default='small', type=str)
    parser.add_argument('-agglo_distance_threshold', help='distance threshold of agglomerative clustering', default=1.4, type=float)
    parser.add_argument('-openai_api_key', help='API KEY published by OpenAI', required=True, type=str)
    parser.add_argument('-openai_org', help='OpenAI Organization', required=True, type=str)
    args = parser.parse_args()

    new_topic_count = None

    # Step 1
    args.n_cluster = get_embedding(args)

    print('--REFINEMENT START--')
    while new_topic_count != 0:
        # Step 2
        print('\nAMBIGUOUS INSTANCE SELETING...')
        select_amb(args)
        # Step 3
        print('\nLLM TOPIC RE-ASSIGNING...')
        predict(args)
        # Step 4
        print('\nSEED TOPIC ADJUSTING...')
        new_topic_count, refined_n_cluster = refine(args)
        args.n_cluster = refined_n_cluster
        print(f'Adding {new_topic_count} topics, totally {refined_n_cluster} topics after refinement.')
        print('-'*50)
    print('--REFINEMENT DONE--')