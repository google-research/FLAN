from flan import task_splits
from flan import tasks
import tensorflow as tf
from tqdm import tqdm
import seqio
import sys
from multiprocessing import Pool
import json

def process_prompt(prompt,response):
    '''Process a FLAN prompt to remove extra tokens'''
    prompt,response = list(map(lambda x:x.numpy().decode(),[prompt,response]))
    prompt = prompt.strip('0').strip('X 1 FLAN').strip()
    response = response.strip()
    return f'{prompt}\n{response}'


def process_single_example(example):
    '''Converts a single prompt to A Decoder only format'''

    inputs_pretokenized = example['inputs_pretokenized']
    targets_pretokenized = example['targets_pretokenized']

    return tf.py_function(
        process_prompt,
        inp=[inputs_pretokenized,targets_pretokenized],
        Tout=tf.string
    )

def prepare_task(task):
    '''Saves a single task data'''
    dataset = seqio.get_mixture_or_task(task).get_dataset(
        sequence_length={'inputs':4096,'targets':4096} # Extranous length to capture all data
    )
    dataset = dataset.map(process_single_example)
    
    data = []
    for i in dataset.as_numpy_iterator():
        data.append(i.decode())
    
    with open(f'./FLANdata/{task}.json','w') as f:
        json.dump(data,f)


if __name__ == '__main__':
    splits = task_splits.generate_superglue_num_tasks_ablation()
    task_split = splits[-1]

    all_tasks = list(task_split.train_tasks)
    all_tasks += task_split.test_tasks

    with Pool(36) as p:
        p.map(prepare_task,all_tasks)

