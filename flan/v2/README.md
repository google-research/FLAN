# The Flan Collection

[**Setup**](#setup) | [**Mixtures**](list-of-mixtures) | [**Run**](how-to-use) | [**Paper**](https://arxiv.org/abs/2301.13688) | [**Citation**](#citation)

The Flan Collection of datasets and data augmentation methods for instruction tuning is generated using the code in this repository. The Flan Collection compiles datasets from Flan 2021, [P3](https://huggingface.co/datasets/bigscience/P3), [Super-Natural Instructions](https://arxiv.org/abs/2204.07705), along with dozens more datasets into one place, formats them into a mix of zero-shot, few-shot and chain-of-thought templates, then mixes these in proportions that are found to achieve strong results on held-out evaluation benchmarks, as reported for Flan-T5 and Flan-PaLM in the [Scaling Flan paper](https://arxiv.org/abs/2210.11416).

## Setup
```
pip install -r flan/v2/requirements.txt
```

## List of Mixtures
We've broken down the Flan Collection into several sub-mixtures. These are "flan" (Flan 2021), "t0" (P3 excluding Flan 2021), "niv2" (Super-Natural Instructions) "cot" (several Chain-of-Thought datasets), and "dialog" (a few new dialog datasets).
Each of these come in 4 varieties of templates: zero-shot prompts with answer options (zsopt), zero-shot prompts with no answer options (zsnoopt), few-shot prompts with answer options (fsopt), and few-shot prompts with no answer options (fsnoopt). Answer options indicate whether for multiple choice classification tasks the set of answers are described in the instruction prompt or not.

mixtures.py contains the following submixture combinations:
```
{flan,t0,cot,dialog,niv2}_{zsopt,zsnoopt,fsopt,fsnoopt}
```

## How to Use
You can import mixtures.py and directly use the mixtures, or combine all the mixtures into a new mixture, for instance:
```
seqio.MixtureRegistry.add(
    'full_cot',
    tasks=[
        ('cot_zsopt', 25),    # mixing weight = 25
        ('cot_fsopt', 25),    # mixing weight = 25
        ('cot_zsnoopt', 25),  # mixing weight = 25
        ('cot_fsnoopt', 25),  # mixing weight = 25
    ])
```

To create the full Flan Collection you will need to create an even mix of each template setting for each submixture, as shown for "cot" above. 
Finally, you can combine the five submixtures into one final mixture, following mixture rates of your choice, or what is recommended in the paper.

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
