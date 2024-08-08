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
For _human_, _fruit fly_, mouse, rat, S. cerevisiae, B. subtilis datasets:

1. The relevant data (~1.5G) can be available at the [Link](https://pan.baidu.com/s/11xFJtqn0ddIl4GUdrm3HvQ?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/`.

3. If you want to train or test the model on different datasets, please modify the parameter settings in the code.

For CAFA dataset:

1. The relevant data (~4.5G) can be available at the [Link](https://pan.baidu.com/s/1EHGFid-cYMtOBcgi3nalbQ?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/`.

Test
---
For _human_, _fruit fly_, mouse, rat, S. cerevisiae, B. subtilis datasets:

`python test.py` used to reproduct the performence recorded in the paper.

For CAFA dataset:

`python test_CAFA3.py` used to reproduct the performence recorded in the paper.

Train
---
For _human_, _fruit fly_, mouse, rat, S. cerevisiae, B. subtilis datasets:

`python main.py`

For CAFA dataset:

`python main_CAFA3.py`

P-value calculation
---
1. The relevant data (~1.5G) can be available at the [Link](https://pan.baidu.com/s/1HeTARs1y-VmJGCiGF17exw?pwd=1234).

2. Unzip the above file to the corresponding directory `./data/Human/`.

3. `python pvalue.py`
