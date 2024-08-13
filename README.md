# [ECCV 2024] Uncertainty Calibration with Energy Based Instance-wise Scaling in the Wild Dataset
 > Official implementation of "[Uncertainty Calibration with Energy Based Instance-wise Scaling in the Wild Dataset](https://arxiv.org/abs/2407.12330) (ECCV '24)"
>[Mijoo Kim](https://sites.google.com/view/mijoo-kim/), and [Junseok Kwon](https://scholar.google.com/citations?user=lwsaTnEAAAAJ&hl=en)
 
[![arXiv](https://img.shields.io/badge/Arxiv-2407.12330-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.12330)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmijoo308%2FEnergy-Calibration&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
<!-- **📢Code will be released soon !** -->

## Abstract
<p align='center'>
<img src='./figures/pipeline.png' width='900'/>
</p>

In this paper, we investigate robust post-hoc uncertainty calibration methods for DNNs within the context of multi-class classification tasks. While previous studies have made notable progress, they still face challenges in achieving robust calibration, particularly in scenarios involving out-of-distribution (OOD). We identify that previous methods lack adaptability to individual input data and struggle to accurately estimate uncertainty when processing inputs drawn from the wild dataset. To address this issue, we introduce a novel instance-wise calibration method based on an energy model. Our method incorporates energy scores instead of softmax confidence scores, allowing for adaptive consideration of DNN uncertainty for each prediction within a logit space.


## Run demo
### (1) Download datasets
- ID dataset (ex. cifar10, cifar100, imagenet )
- OOD dataset
    - Corrupted dataset (ex. cifar10C, cifar100C, imagenetC ) [link](https://github.com/hendrycks/robustness)
    - two Semantic OOD datasets (ex. SVHN, Texture)
        - one for tuning / one for evaluation
```
📦Energy-Calibration
 ├── 📂data
 │   ├──📂cifar10
 │   ├──📂cifar10C
 │   ├──📂SVHN
 │   └──📂dtd
 ├── ...
```

### (2) Environment setup
```
git clone https://github.com/mijoo308/Energy-Calibration.git
cd Energy-Calibration
```

```
conda create -n energycal python=3.8
conda activate energycal
```
```
pip install -r requirements.txt
```


### (3) Run
```
python main.py --gpu --ddp
```

If you already have logit files, you can use the following flags:
- `--id_train_inferenced` : already have all logit files for tuning
- `--ood_train_inferenced` : already have all ood logit files for tuning
- `--test_inferenced` : already have all test logit files for evaluation

### (4) Result
After running the main script, you will see the results in your terminal and the results will be saved in the `result` folder:
```
📦Energy-Calibration
 ├── 📂result
 │   └── 📂cifar10
 │       └── 📂densenet201
 │           └── 📜densenet201_cifar10_{corruption_type}_{severity_level}_result.pkl
 ├── ... 
```

---
## Run using your own classifier
Place the network architecture files in the `./models/` directory and the pretrained weight files in the `./weights/` directory.
```
📦Energy-Calibration
 ├──📂data
 │  ├──📂cifar10
 │  ├──📂cifar10C
 │  ├──📂SVHN
 │  └──📂dtd
 ├──📂models             
 │  └──📜{network}.py    /* place your own network */
 ├──📂result
 ├──📂source
 ├──📂weights
 │  └──📜{network}.pth   /* place your own weight file */
 ├──📜main.py
 ├──...
```
### Run
```
python main.py --gpu --net {network} --weight_path {weight file path} 
```


---


## Acknowledgements

This repository benefits from the following repositories:

- [energy_ood](https://github.com/wetliu/energy_ood)
- [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)
- [temperature_scaling](https://github.com/gpleiss/temperature_scaling)
- [focal_calibration](https://github.com/torrvision/focal_calibration)
- [DensityAwareCalibration](https://github.com/futakw/DensityAwareCalibration)
- [Mix-n-Match-Calibration](https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
- [spline-calibration](https://github.com/kartikgupta-at-anu/spline-calibration)

We greatly appreciate their outstanding work and contributions to the community !


## Citation
If you find this repository useful, please consider citing :
```
@inproceedings{kim2024uncertainty,
    title={Uncertainty Calibration with Energy Based Instance-wise Scaling in the Wild Dataset},
    author={Kim, Mijoo and Kwon, Junseok},
    booktitle={ECCV},
    year={2024}
}
```
