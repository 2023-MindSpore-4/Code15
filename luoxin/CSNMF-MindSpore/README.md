# C-SNMF

This repository is an official MindSpore implementation of our paper "Constraint-Induced Symmetric Nonnegative Matrix Factorization for  Accurate Community Detection". (*Information Fusion, 2023*). [[download](https://www.sciencedirect.com/science/article/abs/pii/S1566253522001300)]


## Prerequisites:

1. MindSpore 2.1.1
2. numpy
3. sklearn


## Dataset

All datasets used in this Paper are follows.

|  Datasets   | Nodes  |  Edges  | Communities |       Description       |
| :---------: | :----: | :-----: | :---------: | :---------------------: |
|    DBLP     | 12,547 | 35,250  |      4      |   DBLP collaboration    |
| Friendster  | 11,023 | 280,755 |     13      |    Friendster online    |
| LiveJournal | 7,181  | 253,820 |     30      |   LiveJournal online    |
|    Orkut    | 11,751 | 270,667 |      5      |      Orkut online       |
|   Youtube   | 11,144 | 36,186  |     40      |     Youtube online      |
|  Polblogs   | 1,490  | 16,718  |      2      | Blogs about US politics |


## Results
### RMSE & MAE with value of lambda
<p align="center">
<img src="imgs/lambda1.png" width="400px" height="300px"/>
</p>

### NMI, ARI and VM with value of lambda
<p align="center">
<img src="imgs/lambda2.png" width="400px" height="300px"/>
</p>

### RMSE, MAE, NMI, ARI and VM with value of mu
<p align="center">
<img src="imgs/mu.png" width="400px" height="300px"/>
</p>

### "Community Detection"
<p align="center">
<img src="imgs/community_detection.png" width="400px" height="300px"/>
</p>


### "RMSE & MAE Convergence"
<p align="center">
<img src="imgs/RMSE_MAE_Convergence.png" width="400px" height="300px"/>
</p>

### "NMI, ARI and VM Convergence"
<p align="center">
<img src="imgs/metric_convergence.png" width="400px" height="300px"/>
</p>

### "minRMSE & minMAE"
<p align="center">
<img src="imgs/result1.png"/>
</p>

### "NMI, ARI and VM"
<p align="center">
<img src="imgs/result2.png"/>
</p>

## Citation

If you find our paper useful in your research, please consider citing:

```
@article{liu2023constraint,
  title={Constraint-Induced Symmetric Nonnegative Matrix Factorization for Accurate Community Detection},
  author={Liu, Zhigang and Luo, Xin and Wang, Zidong and Liu, Xiaohui},
  journal={Information Fusion},
  volume={89},
  pages={588--602},
  year={2023},
  publisher={Elsevier}
}
```

