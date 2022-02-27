# FMG-based Action Recognition
### Official Implementation of '[Optimization of Forcemyography Sensor Placement for Arm Movement Recognition]()'
## Notations

>**Optimization of Forcemyography Sensor Placement for Arm Movement Recognition**<br>
>Xiaohao Xu, Zihao Du, Huaxin Zhang, Ruichao Zhang,
Zihan Hong, Qin Huang, and Bin Han∗, Member, IEEE
>
>**Abstract:**  *How to design an optimal wearable device for human movement recognition is vital to reliable and accurate human-machine collaboration. Previous works mainly fabricate wearable devices heuristically. Instead, this paper raises an academic question:can we design an optimization algorithm to optimize the fabrication of wearable devices such as figuring out the best sensor arrangement automatically? Specifically, this work focuses on optimizing the placement of Forcemyography (FMG) sensors for FMG armbands in application of arm movement recognition. Firstly, the armband is modeled based on graph theory with both sensor signals and connectivity of sensors considered. Then, a Graph-based Armband Modeling Network(GAM-Net) is introduced for arm movement recognition. Afterward, the sensor placement optimization for FMG armbands is formulated and an optimization algorithm with greedy local search is proposed. To study the effectiveness of our optimization algorithm, a dataset for mechanical maintenance tasks using FMG armbands with 16 sensors is collected. Our experiments show that a comparable recognition accuracy with all sensors can be 
maintained even with 4 sensors optimized with our algorithm. Finally, the optimized sensor placement result is verified from a physiological view. This work would like to shed light on the automatic fabrication of wearable devices with downstream tasks, like human biological signal collection and movement recognition, considered.*

## Prerequisites
### Recommended Environment
* Python 3.7
* Pytorch 1.7
* CUDA 10.1

### Depencencies
You can set up the environments by using `$ pip3 install -r requirements.txt`.

### Data Preparation
1. Download our self-collected FMG dataset on mechanical mainte-
nance tasks:https://pan.baidu.com/s/1NYCBs1VkBx20i-INAJZZ2w password:sga8.

2. Place the `original_data_new` inside the working folder.
- ensure the data structure is as below.
~~~~
|-original_data_new
    |-dzh
        |-data_2022-01-07-21-22-13.csv
        ...
        |-data_2022-01-07-21-44-10.csv
    |-hzh
    |-xxh
    |-zhx
    |-zrc
~~~~

3. Preprocess data.
- You can preprocess data by running the script below.
```python
python data_preprocess.py
```






* Tested Subject: **S**
  * Subjects = ["dzh","hzh","xxh","zhx","zrc"] 
* Action: **A**
   * ~~~~
     """
     悬空静止，
     力1静止，力1握5次，力2静止，力2握5次，
     力3静止，力3握5次，力4静止，力4握5次，
     五支张开静止，握拳静止，剪刀手静止，握拳张开
     手腕左右摆动，手臂旋转，力1旋转手臂
     """
     Class_List=["Relaxed_Hand",
                 "Force1_Still","Force1_Motion","Force2_Still","Force2_Motion",
                 "Force3_Still","Force3_Motion","Force4_Still","Force4_Motion",
                 "Finger_abduction","Clench","V_sign","Clench_abduction",
                 "Wrist_swing","Rotating_Arm","Force1_rotation"]
     ~~~~

* Testing numbder I
  * [1,2] 


## Data Preprocessing
Download data:link:https://pan.baidu.com/s/1NYCBs1VkBx20i-INAJZZ2w password:sga8
* Normalize
* Denoise
* Crop with sliding window
  * lengths:[50,100,150,200,250]    
  * strides:[1,3,5,10]

script:
```python
python data_preprocess.py
```

Folder structure:
```latex
|-original_data
    |-dzh
        |-data_2021-04-09-21-01-27.csv
        ...
        |-data_2021-04-09-21-19-17.csv
    |-hzh
    |-xxh
    |-zhx
    |-zrc

|-window_data
    |-L50_s1
        |-S1_A1_I1.npy
        |-S1_A1_I2.npy
        ...
        |-S5_A16_I2.npy   
    |-L50_s3
    ...
    |-L250_s10
```

## Models

### LDA

### LSTM

### ANN

### CNN
