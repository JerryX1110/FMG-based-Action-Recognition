# FMG-based Action Recognition

## Notations

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
