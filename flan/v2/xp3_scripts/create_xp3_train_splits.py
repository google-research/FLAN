import functools
import json
import csv
# pip install -q datasets
import datasets
# git clone -b tr13 https://github.com/Muennighoff/promptsource.git && cd promptsource; pip install -e .
from promptsource.templates import DatasetTemplates
from tqdm import tqdm


# Set to False to use multilingual prompts e.g. 'id' for xcopa/id instead of 'en'
USE_ENGLISH_PROMPTS = True

# Some datasets have test sets with hidden labels which will still compile but only to noise
# e.g. piqa test labels are all [-1] which still works on list indices resulting in 
# noise samples where the label is always the same  
SKIP_PROMPTS = {
    "common_gen": {"test": ["all"]},
    "piqa": {"test": ["all"]},
    "qasc": {"test": ["all"]},
    "imdb": {"unsupervised": ["all"]},
    "glue/qqp": {"test": ["all"]},
    "qasc": {"test": ["all"]},
    "cosmos_qa": {"test": [
        "description_context_question_answer_text", 
        "description_context_question_text",
        "description_context_question_answer_id",
        "context_answer_to_question",
        "context_description_question_answer_text",
        "context_description_question_answer_id",
        "context_question_description_answer_id",
        "context_description_question_text",
        "context_question_description_answer_text",
        "only_question_answer",
        "no_prompt_id",
        "context_question_description_text",
        "no_prompt_text",
        ]},
    "clue/tnews": {"test": ["all"]},
    "clue/csl": {"test": ["all"]},
    "clue/cmrc2018": {"test": ["generate_question", "in_an_exam", "answer_in_the_passage", "answer_following_question", "xp3longcontinue"]},
    "clue/drcd": {"test": ["generate_question", "in_an_exam", "answer_in_the_passage", "answer_following_question", "xp3longcontinue"]},
    "hellaswag": {"test": ["complete_first_then", "Topic of the context", "Open-ended completion", "Randomized prompts template", "Appropriate continuation - Yes or No", "Predict ending with hint", "Open-ended start", "Reversed appropriate continuation - Yes or No", "how_ends", "if_begins_how_continues"]},
}

DS_TO_ENG_PROMPT = {
    "xcopa": "en",
    "Muennighoff/xstory_cloze": "en",
    "Muennighoff/xwinograd": "en",
    'GEM/wiki_lingua': 'en_en', # Contains correct language names
    'xnli': 'en',
    "paws-x": "en",
    "mlqa": "mlqa.en.en",
    "xquad": "xquad.en",
    "khalidalt/tydiqa-primary": "english",
    "khalidalt/tydiqa-goldp": "english",
    "pasinit/xlwic": "en",
    "GEM/xlsum": "english",
    "GEM/BiSECT": "en",
  }

@functools.lru_cache(maxsize=None)
def get_dataset_info(dataset_name):
    info = datasets.get_dataset_infos(dataset_name) # gets a lot of metadata info on the dataset
    return info

def get_dataset_splits(dataset_name, subset_name=None):
    info = get_dataset_info(dataset_name)
    subset_name = subset_name or list(info.keys())[0] # subset name such as 'ak'
    return info[subset_name].splits # provides the relevant splits available

def get_num_examples(dataset_splits):
    return {split: dataset_splits[split].num_examples for split in dataset_splits.keys()}

def get_tasks_splits(ds):

    ### GET DATASET & LANGUAGE ###

    ds_name, subset_name = ds
    dataset_splits = get_dataset_splits(ds_name, subset_name)
    if subset_name == "xlwic_en_zh":
        # Train set is en; val & test are zh
        del dataset_splits["train"]
    elif ds_name == "teven/code_docstring_corpus":
        # Bad quality split
        del dataset_splits["class_level"]

    ### SELECT PROMPTS ###

    if subset_name is None:
        prompt_dataset_name = ds_name
    else:
        subset_name_prompt = subset_name
        if USE_ENGLISH_PROMPTS and ds_name in DS_TO_ENG_PROMPT:
            subset_name_prompt = DS_TO_ENG_PROMPT[ds_name]
        prompt_dataset_name = f"{ds_name}/{subset_name_prompt}"

    prompts = DatasetTemplates(prompt_dataset_name)

    ### PROCESS ###
    splits = []
    for t_name in prompts.all_template_names:
        split_dict = get_num_examples(dataset_splits)
        split_dict['task_name'] = t_name
        split_dict['dataset_name'] = ds_name
        split_dict['subset_name'] = subset_name
        ds_json = f"\'xp3:{ds_name}_{subset_name}_{t_name}\'".replace("/", "_").replace(" ", "_") + f":{split_dict}"
        splits.append(ds_json)

    return splits

def get_train_splits(filename):
    with open(filename, "r") as f:
        reader = csv.reader(f)
        num_rows = sum(1 for row in reader) # count the number of rows

        f.seek(0) #
        next(reader) # skip header
        train_splits = []
        datasets_pbar = tqdm(reader, total=num_rows)
        for ds in datasets_pbar:
            datasets_pbar.set_description(f"Processing {ds[0]}/{ds[1]}")
            train_splits = train_splits + get_tasks_splits(ds)
        return train_splits

TRAIN_SPLITS = get_train_splits("xp3_train_datasets.csv")

with open("xp3_train_splits.txt", "w") as f:
    for s in tqdm(TRAIN_SPLITS, total=len(TRAIN_SPLITS), desc="Writing train splits"):
        f.write(s)
        f.write(",\n")