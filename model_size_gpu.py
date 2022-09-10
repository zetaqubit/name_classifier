"""Tests the GPU ram usage vs training speed for settings like gradient accumulation.

Adapted from
https://huggingface.co/docs/transformers/perf_train_gpu_one
"""
from pynvml import *
from transformers import AutoModelForSequenceClassification


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
print_gpu_utilization()

import numpy as np
from datasets import Dataset


seq_len, dataset_size = 512, 512
dummy_data = {
    "input_ids": np.random.randint(100, 30000, (dataset_size, seq_len)),
    "labels": np.random.randint(0, 1, (dataset_size)),
}
ds = Dataset.from_dict(dummy_data)
ds.set_format("pt")

print('Before any allocations')
print_gpu_utilization()

import torch


torch.ones((1, 1)).to("cuda")
print('After allocation 1x1 Torch tensor.')
print_gpu_utilization()

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased").to("cuda")
print_gpu_utilization()

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

from transformers import TrainingArguments, Trainer, logging

logging.set_verbosity_error()


training_args = TrainingArguments(per_device_train_batch_size=2, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print('Vanilla training (batch size 2).')
print_summary(result)


training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print('Adding gradient accumulation steps=4')
print_summary(result)

training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)
trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print('Adding gradient checkpointing.')
print_summary(result)