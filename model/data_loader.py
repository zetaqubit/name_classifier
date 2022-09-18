import glob
import itertools

import torch as th

from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler, IterableDataset


class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>',
        truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(th.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(th.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


class StreamingDataset(IterableDataset):
  def __init__(self, file_path):
    self._file_path = file_path

  def __iter__(self):
    return iter(open(self._file_path))

class InterleavedDataset(IterableDataset):
  def __init__(self, dir):
    self._files = glob.glob(dir + '/*.txt')
  
  def __iter__(self):
    info = th.utils.data.get_worker_info()
    if info:
      files = self._files[info.id::info.num_workers]
      print(f'In worker {info.id} operating on {" ".join(files)}')
    else:
      files = self._files
      print(f'Operating on {" ".join(files)}')

    iters = [iter(StreamingDataset(file)) for file in files]
    for file, iter_i in zip(files, iters):
      try:
        next(iter_i)
      except Exception as e:
         raise RuntimeError(f'Could not open {file} for reading.') from e
    iters = [itertools.cycle(iter_i) for iter_i in iters]  # make each file infinite

    interleaved = itertools.chain.from_iterable(zip(iters))
    for iter_i in itertools.cycle(iters):
      yield next(iter_i)
  
class FB500Dataset(IterableDataset):
  def __init__(self, dir):
    self._ds = InterleavedDataset(dir)

  def __iter__(self):
    for line in self._ds:
      ex = self.parse_line(line)
      if ex:
        yield ex

  def parse_line(self, line):
    fields = line.split(':')
    if len(fields) < 7: return None
    fn, ln = fields[2], fields[3]
    gender = fields[4]
    birthplace = fields[5]
    currentplace = fields[6]
    if not fn or not ln or not gender or not birthplace or not currentplace: return None
    return f'{fn}:{ln}:{gender}:{birthplace}:{currentplace}'

class GPT2StreamingDataset(IterableDataset):

  def __init__(self, dataset, tokenizer, gpt2_type="gpt2", max_length=768):
    self.tokenizer = tokenizer
    self.dataset = dataset
    self.max_length = max_length
  
  def __iter__(self):
    for txt in self.dataset:
      encodings_dict = self.tokenizer('<|startoftext|>' + txt + '<|endoftext|>',
        truncation=True, max_length=self.max_length, padding="max_length")

      yield (
        th.tensor(encodings_dict['input_ids']),
        th.tensor(encodings_dict['attention_mask']),
      )
