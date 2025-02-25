import time
from openai import OpenAI

# Define a function that adds a delay to a Completion API call
def delayed_completion(OPENAI_API_KEY, OPENAI_ORG, delay_in_seconds: float = 1, max_trials: int = 1, **kwargs):
    client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG)

    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    output, error = None, None
    for _ in range(max_trials):
        try:
            output = client.chat.completions.create(**kwargs)
            break
        except Exception as e:
            error = e
            pass
    return output, error

def prepare_data(dataset_prompt, datum):
    input_txt = datum["input"]
    if input_txt.endswith("\nChoice"):
        input_txt = input_txt[:-7]

    if datum['task'] == 'clinc_domain':
        prefix = f"You are provided with a list of topics and an utterance:\n\nTopics:{datum['options']}\n\nUtterance:{input_txt}\n\n"
    else:
        prefix = f"You are provided with a list of topics and an article:\n\nTopics:{datum['options']}\n\nArticle:{input_txt}\n\n"

    return prefix + dataset_prompt

def post_process(completion, choices):
    content = completion.choices[0].message.content.strip()
    result = []
    return content, result