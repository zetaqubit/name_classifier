import os

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import torch as th

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel


FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name', None,
  'Name of the experiment. Will be used to as a directory name relative to _MODEL_DIR.')

# Parent directory for saving the trained models.
_MODEL_DIR = '/media/14tb/ml/models/zetaqubit/name_classifier/gpt2'


def main(argv):
  exp_dir = os.path.join(_MODEL_DIR, FLAGS.exp_name)
  model, tokenizer = restore_model(exp_dir)
  device = th.device('cuda')
  model.to(device)
  model.eval()

  while True:
    try:
        prompt = input(f'\nName: ')
    except (EOFError, KeyboardInterrupt):
        print()
        break
    texts = generate(model, tokenizer, prompt)
    print('\n'.join(texts))

def generate(model, tokenizer, prompt):
  prompt = '<|startoftext|>' + prompt

  generated = th.tensor(tokenizer.encode(prompt)).unsqueeze(0)
  generated = generated.to('cuda')

  print(generated)

  sampled_outputs = model.generate(
                                  generated, 
                                  #bos_token_id=random.randint(1,30000),
                                  do_sample=True,   
                                  max_length=128,
                                  #top_k=50, 
                                  top_p=0.95, 
                                  num_return_sequences=10,
                                  )
  
  sampled_texts = tokenizer.batch_decode(sampled_outputs, skip_special_tokens=True)
  return sampled_texts


def restore_model(model_dir):
  model = GPT2LMHeadModel.from_pretrained(model_dir)
  tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
  return model, tokenizer


if __name__ == '__main__':
  flags.mark_flag_as_required('exp_name')
  app.run(main)