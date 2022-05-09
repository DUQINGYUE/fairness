# 基于帕累托视角的推荐系统多角度公平性
This is our Tensorflow implementation for the paper:

>杜清月, 黄晓雯, 桑基韬. 基于帕累托效应视角下的推荐系统多角度公平性[J]. 太原理工大学学报, 2022, 53(1):9.

## 简介
提出了一种从帕累托视角解决推荐系统多角度公平性的方法,通过对抗正则化器消除用户嵌入中的敏感属性信息,采用基于曝光的负采样策略提高推荐系统的准确率,从而达到帕累托最优.并且,基于曝光的负采样策略在一定程度上解决物品曝光偏差的问题,保证了物品角度公平性,实现了用户,物品的多角度公平性.实验结果表明,该方法在保证推荐准确率的同时有效提高了用户和物品角度的公平性.
## 引用 
```
@article{杜清月2022基于帕累托效应视角下的推荐系统多角度公平性,
  title={基于帕累托效应视角下的推荐系统多角度公平性},
  author={杜清月 and 黄晓雯 and 桑基韬},
  journal={太原理工大学学报},
  volume={53},
  number={1},
  pages={9},
  year={2022},
}
```
## 环境配置
环境配置如下
1. Pytorch version=1.0
2. scikit-learn
3. tqdm for progress bar
4. pickle
5. json
6. joblib

## 代码实现
实验细节如下：
* MovieLens （以年龄公平性为例）
```
无对抗nohup ipython --pdb -- main_movielens.py --namestr='100 GCMC Comp and Dummy' --use_cross_entropy --num_epochs=200 --test_new_disc --use_1M=True --show_tqdm=True --report_bias=True --valid_freq=5 --use_gcmc=True --num_classifier_epochs=200 --embed_dim=32  --use_age_attr=True --gamma=0 --do_log > age_without.log
```
```
对抗+随机采样nohup ipython --pdb -- main_movielens.py --namestr='100 GCMC Comp and Dummy' --use_cross_entropy --num_epochs=200 --test_new_disc --use_1M=True --show_tqdm=True --report_bias=True --valid_freq=5 --use_gcmc=True --num_classifier_epochs=200 --embed_dim=32  --use_age_attr=True --gamma=1000 --do_log > age_withrandom.log
```
```
对抗+负采样nohup ipython --pdb -- main_movielens.py --namestr='100 GCMC Comp and Dummy' --use_cross_entropy --num_epochs=200 --test_new_disc --use_1M=True --show_tqdm=True --report_bias=True --valid_freq=5 --use_gcmc=True --num_classifier_epochs=200 --embed_dim=32  --use_age_attr=True --gamma=1000 --do_log > age_withneg.log
```
```
对抗+负采样nohup ipython --pdb -- main_movielens.py --namestr='100 GCMC Comp and Dummy' --use_cross_entropy --num_epochs=200 --test_new_disc --use_1M=True --show_tqdm=True --report_bias=True --valid_freq=5 --use_gcmc=True --num_classifier_epochs=200 --embed_dim=32  --use_age_attr=True --gamma=1000 --do_log > age_withneg.log
```
* `评价指标`
  *推荐系统准确率：HR@10，NDCG@10
  * 用户角度公平性：AUC/F1
  * 物品角度公平性e ̅=|e_high/I_high -e_low/I_low |


## 数据集
我们使用MovieLens数据集进行实验验证，用户角度分别考虑了性别，年龄和职业三个敏感属性


|:---:|:---|---:|---:|
|数据集| #用户数 | 物品数| 敏感属性 | 
|MovieLens | #9940 | 6640 | 3 |



* `rating.txt`
  * 用户ID 电影ID 评分数

* `user.txt`
  * 用户ID 性别 年龄 职业
## 致谢
如使用我们数据集可引用以下论文作为参考
```
@article{杜清月2022基于帕累托效应视角下的推荐系统多角度公平性,
  title={基于帕累托效应视角下的推荐系统多角度公平性},
  author={杜清月 and 黄晓雯 and 桑基韬},
  journal={太原理工大学学报},
  volume={53},
  number={1},
  pages={9},
  year={2022},
}
```

