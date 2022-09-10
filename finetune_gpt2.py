"""Script to finetune GPT-2 on the names dataset.
"""

import datetime
import time
import os
import random

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

from model import data_loader

FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name', None,
  'Name of the experiment. Will be used to as a directory name relative to _MODEL_DIR.')
flags.DEFINE_string('dataset_name', None,
  'Path to the dataset to train on. Relative to _DATASET_DIR.')

# Parent directory for saving the trained models.
_MODEL_DIR = '/media/14tb/ml/models/zetaqubit/name_classifier/gpt2'

# Directory containing the dataset.
_DATASET_DIR = '/media/14tb/ml/data/fb_500m/cleaned'


def main(argv):
  print(f'Experiment name: {FLAGS.exp_name}')

  exp_dir = os.path.join(_MODEL_DIR, FLAGS.exp_name)
  if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

  tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>',
    eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

  filepath = os.path.join(_DATASET_DIR, FLAGS.dataset_name)
  with open(filepath) as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

  dataset = data_loader.GPT2Dataset(lines, tokenizer, max_length=128)

  # Split into training and validation sets
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  print('{:>5,} training samples'.format(train_size))
  print('{:>5,} validation samples'.format(val_size))

  batch_size = 16

  # Create the DataLoaders for our training and validation datasets.
  # We'll take training samples in random order. 
  train_dataloader = DataLoader(
              train_dataset,  # The training samples.
              sampler = RandomSampler(train_dataset), # Select batches randomly
              batch_size = batch_size # Trains with this batch size.
          )

  # For validation the order doesn't matter, so we'll just read them sequentially.
  validation_dataloader = DataLoader(
      val_dataset, # The validation samples.
      sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
      batch_size = batch_size # Evaluate with this batch size.
  )


  configuration = GPT2Config.from_pretrained(
      'gpt2',
      output_hidden_states=False,
      pad_token_id=tokenizer.pad_token_id)

  # instantiate the model
  model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

  # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
  # otherwise the tokenizer and model tensors won't match up
  model.resize_token_embeddings(len(tokenizer))

  # Tell pytorch to run this model on the GPU.
  device = th.device("cuda")
  model.cuda()

  set_seed(42)

  epochs = 1
  learning_rate = 5e-4
  warmup_steps = 1e2
  epsilon = 1e-8

  # this produces sample output every 100 steps
  sample_every = 100

  # Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
  optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )
  # Total number of training steps is [number of batches] x [number of epochs]. 
  # (Note that this is not the same as the number of training samples).
  total_steps = len(train_dataloader) * epochs

  # Create the learning rate scheduler.
  # This changes the learning rate as the training loop progresses
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = warmup_steps, 
                                              num_training_steps = total_steps)

  total_t0 = time.time()

  training_stats = []

  model = model.to(device)

  for epoch_i in range(0, epochs):

      # ========================================
      #               Training
      # ========================================

      print("")
      print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
      print('Training...')

      t0 = time.time()

      total_train_loss = 0

      model.train()

      for step, batch in enumerate(train_dataloader):

          b_input_ids = batch[0].to(device)
          b_labels = batch[0].to(device)
          b_masks = batch[1].to(device)

          model.zero_grad()        

          outputs = model(  b_input_ids,
                            labels=b_labels, 
                            attention_mask = b_masks,
                            token_type_ids=None
                          )

          loss = outputs[0]  

          batch_loss = loss.item()
          total_train_loss += batch_loss

          # Get sample every x batches.
          if step % sample_every == 0 and not step == 0:

              elapsed = format_time(time.time() - t0)
              print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.   Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))

              model.eval()

              sample_outputs = model.generate(
                                      bos_token_id=random.randint(1,30000),
                                      do_sample=True,   
                                      top_k=50, 
                                      max_length = 200,
                                      top_p=0.95, 
                                      num_return_sequences=1
                                  )
              for i, sample_output in enumerate(sample_outputs):
                    print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
              
              model.train()

          loss.backward()

          optimizer.step()

          scheduler.step()

      # Calculate the average loss over all of the batches.
      avg_train_loss = total_train_loss / len(train_dataloader)       
      
      # Measure how long this epoch took.
      training_time = format_time(time.time() - t0)

      print("")
      print("  Average training loss: {0:.2f}".format(avg_train_loss))
      print("  Training epoch took: {:}".format(training_time))
          
      # ========================================
      #               Validation
      # ========================================

      print("")
      print("Running Validation...")

      t0 = time.time()

      model.eval()

      total_eval_loss = 0
      nb_eval_steps = 0

      # Evaluate data for one epoch
      for batch in validation_dataloader:
          
          b_input_ids = batch[0].to(device)
          b_labels = batch[0].to(device)
          b_masks = batch[1].to(device)
          
          with th.no_grad():        

              outputs  = model(b_input_ids, 
  #                            token_type_ids=None, 
                              attention_mask = b_masks,
                              labels=b_labels)
            
              loss = outputs[0]  
              
          batch_loss = loss.item()
          total_eval_loss += batch_loss        

      avg_val_loss = total_eval_loss / len(validation_dataloader)
      
      validation_time = format_time(time.time() - t0)    

      print("  Validation Loss: {0:.2f}".format(avg_val_loss))
      print("  Validation took: {:}".format(validation_time))

      # Record all statistics from this epoch.
      training_stats.append(
          {
              'epoch': epoch_i + 1,
              'Training Loss': avg_train_loss,
              'Valid. Loss': avg_val_loss,
              'Training Time': training_time,
              'Validation Time': validation_time
          }
      )

  print("")
  print("Training complete!")
  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

  print("Saving model to %s" % exp_dir)
  save_model(tokenizer, model, exp_dir)


# Set the seed value all over the place to make this reproducible.
def set_seed(seed_val):
  random.seed(seed_val)
  np.random.seed(seed_val)
  th.manual_seed(seed_val)
  th.cuda.manual_seed_all(seed_val)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def save_model(tokenizer, model, output_dir):
  # Save a trained model, configuration and tokenizer using `save_pretrained()`.
  # They can then be reloaded using `from_pretrained()`
  model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
  model_to_save.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
  flags.mark_flag_as_required('exp_name')
  flags.mark_flag_as_required('dataset_name')
  app.run(main)
