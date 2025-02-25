import os
import argparse
import json
import openai
from tqdm import tqdm
from tools import delayed_completion, prepare_data, post_process
from load_save_data import load_predict_data, save_predict_data, load_prompts


def predict(args):
    data, pred_path = load_predict_data(args.dataset, args.scale, args.model_name, args.n_cluster)
    print("Save in: ", pred_path)

    task_prompt = load_prompts(args.dataset)

    for d in data:
        if 'prepared' not in d:
            d['prepared'] = prepare_data(task_prompt, d)

    for idx, datum in tqdm(enumerate(data), total=len(data)):
        if 'prediction' in datum:
            continue

        messages = [
            {"role": "user", "content": datum['prepared']}
        ]
        completion, error = delayed_completion(args.openai_api_key, args.openai_org, delay_in_seconds=1, max_trials=10, model=args.model_name, messages=messages, max_tokens=10, temperature=0)
        
        if completion is None:
            print(f"Saving data after {idx + 1} inference.")
            save_predict_data(pred_path, data)
            print(idx)
            print(error)
            continue
        else:
            content, results = post_process(completion, datum['options'])
            data[idx]['content'] = content
            data[idx]['prediction'] =  results

        if idx % 50 == 0 and idx > 0:
            print(f"Saving data after {idx} inference.")
            save_predict_data(pred_path, data)

    save_predict_data(pred_path, data)
    print("Done")