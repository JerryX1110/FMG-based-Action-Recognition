from time import time_ns
import numpy as np
import pandas as pd
import utils
import matplotlib.pyplot as plt
from model_config import build_args

def model_performance():
    file_dir="./output/model_performance"
    model_name="LDA"
    result=[]
    for model_name in ["LDA","ANN","CNN","LSTM"]:
        print(model_name)
        args=build_args(model_name)
        

        cm_path=file_dir+"/{}_CM.npy".format(model_name)
        acc_path=file_dir+"/{}_ACC.npy".format(model_name)
        cm=np.load(cm_path)
        acc=np.load(acc_path)
        print(acc)
        result.append(acc)
        # CM=np.rint(cm.mean(axis=0)) #rint取整
        # print(CM)
        # title="Confusion matrix of "+args.model_name
        # utils.v_confusion_matrix(CM,args.part_actions,title=title,save_path=save_dir+"/{}_CM.png".format(args.model_name))
        # utils.v_confusion_matrix(CM,args.part_actions,title=title,save_path=save_dir+"/{}_CM.pdf".format(args.model_name))
        # plt.show()

    result=pd.DataFrame(result,index=["LDA","ANN","CNN","LSTM"],columns=["S1","S2","S3","S4","S5"])
    result["mean"]=result.mean(axis=1)
    result["std"]=result.std(axis=1)
    print(result)

    result.to_csv(file_dir+"/model_performance.csv")

def used_time():
    file_dir="./output/model_usedtime"
    model_name="LDA"
    result=[]

    for model_name in ["LDA","ANN","CNN","LSTM"]:
        print(model_name)
        args=build_args(model_name)
        time_path=file_dir+"/{}_Time.npy".format(model_name)
        usedtime=np.load(time_path)
        result.append(usedtime)

    result=pd.DataFrame(result,index=["LDA","ANN","CNN","LSTM"],columns=["S1","S2","S3","S4","S5"])
    result["mean"]=result.mean(axis=1)
    result["std"]=result.std(axis=1)
    print(result)
    result.to_csv(file_dir+"/model_usedtime.csv")



if __name__=="__main__":
    used_time()



    