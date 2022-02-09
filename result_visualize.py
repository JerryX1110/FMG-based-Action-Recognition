import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 
plt.rcParams['axes.unicode_minus'] = False
from pylab import mpl
mpl.rcParams['font.size'] = 12
from model_config import build_args
import os


def temporal_curve():
    data_root="./original_data"
    subjects=["dzh","hzh","xxh","zhx","zrc"]
    subject=subjects[0]
    path=os.path.join(data_root,subject)
    input_file=os.listdir(path)[26]#13
    input_dir=os.path.join(path,input_file)
    print(input_dir)
    filename=input_file

    df=pd.read_csv(input_dir,header=None).T
    df=df.iloc[150:600].reset_index(drop=True)
    n_channel=df.shape[1]
    df.columns=['CH{}'.format(i+1) for i in range(n_channel)]
    # df=df.rolling(window=10).mean()

    # fig,ax=plt.subplots()
    plt.figure(figsize=(12,8))
    grid = plt.GridSpec(3, 3, wspace=0.2,hspace=0.4)
    ax1=plt.subplot(grid[1:,0:3])
    
    df.plot(ax=ax1)
    ax1.set_xlim(0,df.shape[0])
    ax1.set_xlabel('Time/ms')
    ax1.set_ylabel('Voltage/mV')
    ax1.legend(bbox_to_anchor=(1,1),frameon=True)
    
    
    #Ledamap
    sensor_loc_list=np.array([3,11,2,12,10,9,15,16,8,5,7,1,4,6,13,14])-1
    group_list=[sensor_loc_list[:6],sensor_loc_list[6:12],sensor_loc_list[12:]]
    band=["I","II","III"]
    for i in range(3):
        band_data=df.iloc[:,group_list[i]]
        # print(band_data.info())
        ax=plt.subplot(grid[0,i],polar=True)
        values=band_data.mean()
        angles=np.linspace(0,2*np.pi,len(values),endpoint=False)
        values=np.concatenate((values,[values[0]]))
        angles=np.concatenate((angles,[angles[0]]))
        ax.plot(angles,values,'o-',color='#ff0000',linewidth=2)
        ax.fill(angles,values,color='#ff0000',alpha=0.25)
        ax.set_thetagrids(angles*180/np.pi,band_data.columns)
        if i!=2:
            ax.set_xlabel("\nArmband {}".format(band[i]),labelpad=10)
        else:
            ax.set_xlabel("Armband {}".format(band[i]),labelpad=10)
    
    # plt.suptitle(input_dir)
    plt.savefig(os.path.join("./figure","signal_sample_new.pdf"),bbox_inches="tight",dpi=300)

def model_performance():
    result=[]
    for model_name in ["LDA","LSTM","ANN","CNN","GCN"]:
        print(model_name)
        args=build_args(model_name)
        file_dir=os.path.join(args.output_root,"model_performance")

        cm_path=file_dir+"/{}_CM.npy".format(model_name)
        acc_path=file_dir+"/{}_ACC.npy".format(model_name)
        cm=np.load(cm_path)
        acc=np.load(acc_path)
        print(acc)
        result.append(acc)
        CM=np.rint(cm.mean(axis=0)) #rint取整
        print(CM)
        title="Confusion Matrix of "+args.model_name
        utils.v_confusion_matrix(CM,args.part_actions,title=title,save_path=file_dir+"/{}_CM.png".format(args.model_name))
        utils.v_confusion_matrix(CM,args.part_actions,title=title,save_path=file_dir+"/{}_CM.pdf".format(args.model_name))
        

    result=pd.DataFrame(result,index=["LDA","LSTM","ANN","CNN","GCN"],columns=["S1","S2","S3","S4","S5"])
    result["mean"]=result.mean(axis=1)
    result["std"]=result.std(axis=1)
    print(result)

    # result.to_csv(file_dir+"/model_performance.csv")

def used_time():
    result=[]

    for model_name in ["LDA","LSTM","ANN","CNN","GCN"]:
        print(model_name)
        args=build_args(model_name)
        file_dir=os.path.join(args.output_root,"model_usedtime")

        time_path=file_dir+"/{}_Time.npy".format(model_name)
        usedtime=np.load(time_path)
        result.append(usedtime)

    result=pd.DataFrame(result,index=["LDA","LSTM","ANN","CNN","GCN"],columns=["S1","S2","S3","S4","S5"])
    result["mean"]=result.mean(axis=1)
    result["std"]=result.std(axis=1)
    print(result)
    result.to_csv(file_dir+"/model_usedtime_cpu.csv")



if __name__=="__main__":
    temporal_curve()
    # model_performance()
    # used_time()
    plt.show()



    