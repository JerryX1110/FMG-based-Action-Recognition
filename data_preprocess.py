import os
from tqdm import tqdm
from shutil import copyfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False

from pylab import mpl
mpl.rcParams['font.size'] = 12

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def plot_data(df,filename,df_act=None,mode="separate",save=False):
    if mode == "holdon":
        fig,ax=plt.subplots()
        df.plot(ax=ax,legend=True)
        fig.set_size_inches((10,3))
        ax.set_xlim(0,df.shape[0])
        plt.xlabel('Time/ms')
        plt.title(filename)

    if mode=="separate":
        df_mean=df.mean(axis=1)
        fig, ax=plt.subplots(8,2,sharex=True)
        plt.suptitle(filename)
        ax=ax.reshape(-1)
        for i in range(len(ax)):
            
            df.iloc[:,i].plot(ax=ax[i],color="black")
            if not df_act == None:
                ax[i].scatter(df_act.index,df_act.iloc[:,i],color='green',s=30,alpha=0.3)
            df_mean.plot(ax=ax[i],color="r")

            ax[i].set_ylim(0,1)
            ax[i].set_xlim(0,df.shape[0])
            ax[i].set_ylabel("CH{}".format(i))
            ax[i].set_xlabel('Time/ms')
            ax[i].set_yticks([0,1],[0,1])
            ax[i].tick_params(direction='in')
        fig.set_size_inches((12,7))
        if save:
            fig_ouputpath=os.path.join("./figure","Temporal_curve_new")
            make_dir(fig_ouputpath)
            plt.savefig(os.path.join(fig_ouputpath,filename+".png"),bbox_inches="tight",dpi=300)
            plt.cla()
            plt.close("all")

    
def make_data():

    # """
    # 悬空静止，
    # 力1静止，力1握5次，力2静止，力2握5次，
    # 力3静止，力3握5次，力4静止，力4握5次，
    # 五支张开静止，握拳静止，剪刀手静止，握拳张开
    # 手腕左右摆动，手臂旋转，力1旋转手臂
    # """
    # CLASS_LIST=["Relaxed_Hand",
    #             "Force1_Still","Force1_Motion","Force2_Still","Force2_Motion",
    #             "Force3_Still","Force3_Motion","Force4_Still","Force4_Motion",
    #             "Finger_abduction","Clench","V_sign","Clench_abduction",
    #             "Wrist_swing","Rotating_Arm","Force1_rotation"]
    # input_dir="./original_data"
    # output_path="D://数据集/window_data"
    # repeat=2
    

    CLASS_LIST=["Keep_Still","Screw_with_Screwdriver","Lift_a_Bottle","Screw_the_bottle_Cap",
                "use_Glue_Gun_Still","use_Glue_Gun","Lift_up_Object","KeyBoarding","Cut_the_Rope",
                "Tie_a_Rope","Rub_a_Surface_with_sandpaper","WriTing","Wipe_the_Table"]
    CLASS_LIST_SHORT=["KS","SS","LB","SC"
                "GGS","GG","LO","KB","CR",
                "TR","RS","WR","WT"]
    input_dir="./original_data_new"
    output_path="D://数据集/window_data_new"
    repeat=3


    class_dict={action_cls:idx for idx, action_cls in enumerate(CLASS_LIST)}
    subjects=["dzh","hzh","xxh","zhx","zrc"]
    
    # subjects=[subjects[0]]
    # CLASS_LIST=[CLASS_LIST[4]]
    for i,subject in tqdm(enumerate(subjects),total=len(subjects)):
        # i=0
        sub_path=os.path.join(input_dir,subject)
        sub_files=os.listdir(sub_path)
        for j,act in tqdm(enumerate(CLASS_LIST),total=len(CLASS_LIST)):
            # j=4
            for r in range(repeat):
                pre_process(os.path.join(sub_path,sub_files[repeat*j+r]),output_path,"S{}_A{}_I{}".format(i+1,j+1,r+1))
                
            

def pre_process(file_path,savedir,filename):
    df=pd.read_csv(file_path,header=None).T #[T,C]

    #删去前后50个数据
    df=df.iloc[50:-50].reset_index(drop=True)
    n_channel=df.shape[1]
    df.columns=['CH{}'.format(i+1) for i in range(n_channel)]

    #min_max Normalization
    df_norm= (df-df.min())/(df.max()-df.min())
    
    #window smooth
    df_roll=df_norm.rolling(window=10).mean()
    # plot_data(df_roll,filename,save=True)

    #rooling window
    L_wins=[50,100,150,200,250]
    strides=[1,3,5,10]
    for L_win in L_wins:
        for stride in strides:

    # L_win=250
    # stride=10

            window_data=get_window_data(df_roll,L_win,stride)
            path=os.path.join(savedir,"L{}_s{}".format(L_win,stride))
            make_dir(path)
            np.save(os.path.join(path,filename+".npy"),window_data) 



def get_window_data(data,L_win,stride):#data:[T,C]
    data=data.values
    data[np.isnan(data)] = 0
    window_data=[]
    start=0
    while (start+L_win-1) < data.shape[0]:
        window_data.append(data[start:start+L_win,:])
        start += stride
    return window_data




if __name__ == "__main__":
    
    make_data()
    # plt.show()

    # from LDA import window_stride_analysis
    # window_stride_analysis()

    



    

        

