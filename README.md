# [ECCV 2024] Uncertainty Calibration with Energy Based Instance-wise Scaling in the Wild Dataset
 > Official implementation of "[Uncertainty Calibration with Energy Based Instance-wise Scaling in the Wild Dataset](https://arxiv.org/abs/2407.12330) (ECCV '24)"
>[Mijoo Kim](https://sites.google.com/view/mijoo-kim/), and [Junseok Kwon](https://scholar.google.com/citations?user=lwsaTnEAAAAJ&hl=en)
 
[![arXiv](https://img.shields.io/badge/Arxiv-2407.12330-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.12330)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmijoo308%2FEnergy-Calibration&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)\
**📢Code will be released soon !**

## Abstract
<p align='center'>
<img src='./figures/pipeline.png' width='900'/>
</p>

In this paper, we investigate robust post-hoc uncertainty calibration methods for DNNs within the context of multi-class classification tasks. While previous studies have made notable progress, they still face challenges in achieving robust calibration, particularly in scenarios involving out-of-distribution (OOD). We identify that previous methods lack adaptability to individual input data and struggle to accurately estimate uncertainty when processing inputs drawn from the wild dataset. To address this issue, we introduce a novel instance-wise calibration method based on an energy model. Our method incorporates energy scores instead of softmax confidence scores, allowing for adaptive consideration of DNN uncertainty for each prediction within a logit space.

