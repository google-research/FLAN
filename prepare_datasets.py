from flan import task_splits
from flan import tasks
import seqio
import sys

if __name__ == '__main__':

    # sys.stdout = open('out.txt','w')
    sys.stderr = open('err.txt','w')
    splits = task_splits.generate_superglue_num_tasks_ablation()

    for task_split in splits:
        for train_task in task_split.train_tasks:
            dataset = seqio.get_mixture_or_task(train_task).get_dataset(
                sequence_length={'inputs':4096,'targets':4096}
            )
            curr = 5
            for ex in dataset.as_numpy_iterator():
                
                inputs_pretokenized = ex['inputs_pretokenized'].decode('ascii')
                targets_pretokenized = ex['targets_pretokenized'].decode('ascii')
                print(targets_pretokenized)
                break
            break
        break

