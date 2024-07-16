# MIF2GO
Code for paper "MIF2GO: Annotating Protein Functions via Fusing Multiple Biological Modalities"
---

Dependencies
---

python == 3.7.16

pytorch == 1.13.1

PyG (torch-geometric) == 2.3.1

sklearn == 1.0.2

scipy == 1.7.3

numpy == 1.21.5

Data preparation (Password:1234)
---
1. The relevant data (~1.5G) can be available at the [Link](https://pan.baidu.com/s/11xFJtqn0ddIl4GUdrm3HvQ?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/`.

3. If you want to train or test the model on different datasets, please modify the parameter settings in the code.

Test
---

`python test.py` used to reproduct the performence recorded in the paper.

Train
---
`python main.py`

P-value calculation
---
1. The relevant data (~1.5G) can be available at the [Link](https://pan.baidu.com/s/1qzDFYrz0ms_8_rDnQLTu_g?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/Human/`.

3. `python pvalue.py`
