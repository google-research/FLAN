# The FLAN Instruction Tuning Repository

[**Original Flan (2021)**](#flan-2021) | [**The Flan Collection (2022)**](https://github.com/google-research/FLAN/tree/main/flan/v2) | 
[**Tasks**](#task-description) | [**Flan 2021 Citation**](#flan-2021-citation) | [**License**](#license)

This repository contains code to generate instruction tuning dataset collections. The first is the original Flan 2021, documented in [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652), and the second is the expanded version, called the Flan Collection, described in [The Flan Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688) and used to produce [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) and [Flan-PaLM](https://arxiv.org/abs/2210.11416).

Finetuned Language Models are Zero-Shot Learners, Wei et. al., 2021.

## Flan 2021
To generate the Flan 2021 data as Seqio mixtures, first install the relevant `requirements.txt` then use [mixtures.py](https://github.com/google-research/FLAN/blob/main/flan/mixtures.py).

## Flan 2021 Citation
Please cite the following if you found Flan 2021 useful in your research.
```
@inproceedings{weifinetuned,
  title={Finetuned Language Models are Zero-Shot Learners},
  author={Wei, Jason and Bosma, Maarten and Zhao, Vincent and Guu, Kelvin and Yu, Adams Wei and Lester, Brian and Du, Nan and Dai, Andrew M and Le, Quoc V},
  booktitle={International Conference on Learning Representations}
}
```

## License
The code in this repository is licensed according to the [LICENSE](LICENSE) file.

## Contact Us
To contact us feel free to create an Issue in this repository, or email the respective authors that contributed to this code base: Jason Wei for the [Flan 2021](https://arxiv.org/abs/2109.01652) paper, Le Hou for the [Scaling Flan](https://arxiv.org/abs/2210.11416) paper, and Shayne Longpre for the [Flan Collection](https://arxiv.org/abs/2301.13688).
