import logging
import os
import sys
import traceback
from dotenv import load_dotenv
from flask import Flask, g, jsonify, request
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from peft import PeftModelForCausalLM
from read import read_config
import torch
import time


load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.DEBUG
)
app_name = 'gpt4all-api'
logger = logging.getLogger(app_name)

app = Flask(app_name)

port = os.environ['PORT']


def generate(tokenizer, prompt, model, config):
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=config['max_new_tokens'], temperature=config['temperature'])
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return decoded[len(prompt):]


def setup_model(config):
    model = AutoModelForCausalLM.from_pretrained(config['model_name'], device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    added_tokens = tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})

    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))

    if config['lora']:
        model = PeftModelForCausalLM.from_pretrained(model, config['lora_path'], device_map='auto', torch_dtype=torch.float16, offload_folder='/tmp/offload')
        model.to(dtype=torch.float16)

    print(f'Mem needed: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB')

    return model, tokenizer


dirname = os.path.basename(os.path.dirname(__file__))
config = read_config(dirname + '/configs/generate/generate.yaml')

model, tokenizer = setup_model(config)


@app.route('/api/v0.1.0/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt')
    if prompt is None:
        error = {
            'error': {
                'type': 'BadRequest',
                'message': '`prompt` is required',
            },
            'status': 400
        }
        return error, 400
    
    logger.debug('Generating...')
    start = time.time()
    try:
        generation = generate(tokenizer, prompt, model, config)
        logger.debug(f'Done in {time.time() - start:.2f}s')
        results = {
            'generation': generation
        }

        return results, 200

    except Exception as e:
        logger.error(f'General exception')
        print('-' * 80)
        traceback.print_exc(file=sys.stdout)
        print('-' * 80)
        error = {
            'error': {
                'type': type(e).__name__,
                'message': str(e),
            },
            'status': 500
        }
        return error, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
