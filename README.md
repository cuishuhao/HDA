# HDA implemented in PyTorch
code release for "Heuristic Domain Adaptation"(NIPS2020)

## Poster

<div>
<img src="./doc/poster.pdf" width="800">
<div>

  
## Enviroment
- pytorch = 1.3.0
- torchvision = 0.4.1
- numpy = 1.17.2
- pillow = 6.2.0
- python3.7
- cuda10

To install the required python packages, run

```
pip install -r requirements.txt
```

## dataset

Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

Domainnet dataset can be found [here](http://ai.bu.edu/M3SDA/).

The training of HDA could be utilized by changing the path of the dataset, such as the txt files in [data/UDA_officehome/Art.txt](./data/UDA_officehome/Art.txt).

Also, the txt files for SSDA and MSDA should be compressed.

```
cd data
unzip data.zip
```

## training
###UDA on Office-Home
```
bash scripts/run_uda.sh
```

###MSDA on Domainnet
```
bash scripts/run_msda.sh
```

###SSDA on Domainnet
```
bash scripts/run_ssda.sh
```

## Citation
If you use this code for your research, please consider citing:
```
@inproceedings{cui2020hda,
title={Heuristic Domain Adaptation},
author={Cui, Shuhao and Jin, Xuan, and Wang, Shuhui and He, Yuan and Huang, Qingming},
booktitle={Advances in Neural Information Processing Systems},
year={2020}
}
```

## Contact                                                                                                                                                                       
If you have any problem about our code, feel free to contact
- hassassin1621@gmail.com

or describe your problem in Issues.

