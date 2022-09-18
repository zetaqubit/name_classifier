import itertools

import pytest
import torch as th

from model import data_loader

def test_reads_lines():
  ds = data_loader.StreamingDataset('test_data/a.txt')
  lines = list(ds)
  assert lines == ['A1\n', 'A2\n', 'A3\n', 'A4\n']
    
def test_interleaved_loads_files():
  ds = data_loader.InterleavedDataset('test_data')
  dl = th.utils.data.DataLoader(ds, batch_size=1, num_workers=2)
  items = list(itertools.islice(dl, 0, 4))
  assert items == [['A1\n'], ['B1\n'], ['A2\n'], ['B2\n'],]

def test_fb500():
  ds = data_loader.FB500Dataset('test_data/fb500')
  dl = th.utils.data.DataLoader(ds, batch_size=1, num_workers=2)
  items = list(itertools.islice(dl, 0, 4))
  assert items == [
    ['John Doe, male, born in NYC, NY, living in Albany, NY'],
    ['Hui Li, male, born in Shanghai, living in Beijing'],
    ['Jane Doe, female, born in Orlando, Florida, living in Austin, Texas'],
    ['Elain Gu, female, born in Hong Kong, living in Xi\'an'],
  ]

def test_fb500_batched():
  ds = data_loader.FB500Dataset('test_data/fb500')
  dl = th.utils.data.DataLoader(ds, batch_size=2, num_workers=2)
  items = list(itertools.islice(dl, 0, 2))
  assert items == [
    ['John Doe, male, born in NYC, NY, living in Albany, NY',
     'Jane Doe, female, born in Orlando, Florida, living in Austin, Texas'],
    ['Hui Li, male, born in Shanghai, living in Beijing',
     'Elain Gu, female, born in Hong Kong, living in Xi\'an'],
  ]
