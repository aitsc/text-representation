# Introduction
This code is the implementation of the paper "Coherence-Based Distributed Document Representation Learning for Scientific Documents".

# Run
## Installation environment
```shell
conda create -n ctpe python=3.6
conda activate ctpe
git clone https://github.com/aitsc/text-representation.git
cd text-representation
pip install https://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
pip install allennlp==0.4.3
pip install -r requirements.txt
```

## Download data
- Link: https://pan.baidu.com/s/1EEJk0_P55Ov5ReXsmyVZPA Password: rkh0
- Place the `data` folder in the code directory.

## Test
- Modify the `ap` and model address in the code `_av_CTE.py`.
- `python _av_CTE.py`

# Citation
If you find this code useful, please cite the following paper:
```
@article{tan2022coherence,
  title = {Coherence-Based Distributed Document Representation Learning for Scientific Documents},
  author = {Tan, Shicheng and Zhao, Shu and Zhang, Yanping},
  journal = {arXiv},
  year = {2022},
  type = {Journal Article}
}
```
