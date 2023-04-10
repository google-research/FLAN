# The Flan Collection

[**Setup**](#setup) | [**Mixtures**](list-of-mixtures) | [**Run**](how-to-use) | [**Flan Collection Paper**](https://arxiv.org/abs/2301.13688) | [**Citation**](#citation)

The Flan Collection of datasets and data augmentation methods for instruction tuning is generated using the code in this repository. The Flan Collection compiles datasets from Flan 2021, [P3](https://huggingface.co/datasets/bigscience/P3), [Super-Natural Instructions](https://arxiv.org/abs/2204.07705), along with dozens more datasets into one place, formats them into a mix of zero-shot, few-shot and chain-of-thought templates, then mixes these in proportions that are found to achieve strong results on held-out evaluation benchmarks, as reported for Flan-T5 and Flan-PaLM in the [Scaling Flan paper](https://arxiv.org/abs/2210.11416) and [Flan Collection paper](https://arxiv.org/abs/2301.13688).

## Setup
```
bash setup.sh
```

## List of Mixtures
We've broken down the Flan Collection into several sub-mixtures. These are "flan" (Flan 2021), "t0" (P3 excluding Flan 2021), "niv2" (Super-Natural Instructions) "cot" (several Chain-of-Thought datasets), and "dialog" (a few new dialog datasets).
Each of these come in multiple varieties of templates: zero-shot prompts with answer options (zsopt), zero-shot prompts with no answer options (zsnoopt), few-shot prompts with answer options (fsopt), and few-shot prompts with no answer options (fsnoopt). Answer options indicate whether for multiple choice classification tasks the set of answers are described in the instruction prompt or not. These submixtures are instantiated in `flan/v2/mixtures.py`.

## How to Use
You can import `flan/v2/mixtures.py` and directly use the mixtures, or combine all the mixtures into new mixtures. See `flan/v2/run_example.py` for examples of different mixtures you can create and run, as well as the mixture for the Flan 2022 Collection.

```
PYTHONPATH=. python flan/v2/run_example.py
```

NB #1: These scripts download and process dozens of GBs of data, which is usually not feasible in a single run. We recommend starting with submixtures like `cot_submix`, `flan2021_submix`, `dialog_submix`, `t0_submix` and `niv2_submix`, as shown in `flan/v2/run_example.py`. If you plan to use Seqio/T5X for training then we recommend caching the datasets, following these [instructions](https://github.com/google/seqio#optional-offline-caching). If not, you can use the above script to collect the data as raw text/json.

NB #2: Unfortunately a couple datasets from the Flan Collection, used to train Flan-T5 and Flan-PaLM, could not be included in these generation scripts due to legal constraints. Those are Dr Repair datasets (https://github.com/michiyasunaga/DrRepair), Deepmind Code Contests (https://github.com/deepmind/code_contests), and Task Master (https://github.com/google-research-datasets/Taskmaster). As a result, we have no Program Synthesis submixture, as described in the paper. However, these datasets comprise a small minority of overall training examples and their exclusion should have negligible effect on results reported in the paper.

NB #3: If you hit checksum errors from Tensorflow Datasets (TFDS), try updating to the latest `tfds-nightly` then loading the problematic dataset directly with [tfds.load(...)](https://www.tensorflow.org/datasets/api_docs/python/tfds/load). As dataset requirements may change over time (e.g. approvals), occassionally some datasets may no longer be available or require manual download, as suggested [here](https://github.com/google-research/FLAN/issues/37#issuecomment-1479810887).

## Citation
Please cite the following if you found The Flan Collection, our [paper](https://arxiv.org/abs/2301.13688), or these resources useful.
```
@article{longpre2023flan,
  title={The Flan Collection: Designing Data and Methods for Effective Instruction Tuning},
  author={Longpre, Shayne and Hou, Le and Vu, Tu and Webson, Albert and Chung, Hyung Won and Tay, Yi and Zhou, Denny and Le, Quoc V and Zoph, Barret and Wei, Jason and others},
  journal={arXiv preprint arXiv:2301.13688},
  year={2023}
}
```

## Contact Us
To contact us feel free to create an issue in this repository or email the authors in the paper.
