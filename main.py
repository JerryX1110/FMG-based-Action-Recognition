from model import LSTMnet,CNNnet,ANNnet,GCNnet
from LDA import LDAmodel
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from dataset import FMGdataset
from tqdm import tqdm
from model_config import build_args
import matplotlib.pyplot as plt
import utils
import json
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' 

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device=torch.device("cpu")
    print("using {} device.".format(device))
    args.device=device

    if args.model_name == "LSTM":
        model=LSTMnet(in_dim=len(args.channels),hidden_dim=args.hidden_dim,n_layer=args.n_layer,n_class=args.n_class)
    elif args.model_name == "ANN":
        model=ANNnet(in_dim=len(args.channels),hidden_dim=args.hidden_dim,n_layer=args.n_layer,n_class=args.n_class)
    elif args.model_name == "CNN":
        model=CNNnet(in_dim=len(args.channels),hidden_dim=args.hidden_dim,n_layer=args.n_layer,n_class=args.n_class)
    elif args.model_name == "GCN":
        model=GCNnet(input_dim=args.L_win,hidden_dim=args.hidden_dim,output_dim=args.n_class)
        adj=utils.get_adjmatrix()
        i = torch.from_numpy(adj[0]).long().to(device)
        v = torch.from_numpy(adj[1]).to(device)
        adj = torch.sparse.FloatTensor(i.t(), v, adj[2]).float().to(device)
    else:
        print("Model's name is not in the list!");return -1
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss(reduction='none')

    #train_val
    # args.subindex=[0]
    print("TrainVal_subjects_index:{}".format(args.subindex))
    train_dataset=FMGdataset(args,test_ratio=args.test_ratio,phase="train")
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=1)
    test_dataset=FMGdataset(args,test_ratio=args.test_ratio,phase="test")
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=1)
    #test
    # tempsub=list(range(5))
    # tempsub.remove(args.subindex[0])
    # args.subindex=tempsub
    # print("Test_subjects_index:{}".format(args.subindex))
    # CS_test_dataset=FMGdataset(args,test_ratio=1,phase="test")
    # CS_test_loader=torch.utils.data.DataLoader(CS_test_dataset,batch_size=1,shuffle=False,num_workers=1)

    if args.mode == "train":
        
        save_path="./models"
        utils.make_dir(save_path)
        best_acc = 0.0
        best_cs_acc = 0.0
        for epoch in range(args.epochs):
            if args.model_name=="GCN":
                utils.train(epoch,model,train_loader,optimizer,criterion,device,adj)
                _,acc,_=utils.test(epoch,model,test_loader,criterion,args.n_class,device,adj)
            else:
                utils.train(epoch,model,train_loader,optimizer,criterion,device)
                _,acc,_=utils.test(epoch,model,test_loader,criterion,args.n_class,device)
            
            if acc > best_acc:
                best_acc=acc
                torch.save(model.state_dict(),os.path.join(save_path,"best{}.pth".format(args.model_name)))
            cs_acc=0
            # _,cs_acc=utils.test(epoch,model,CS_test_loader,criterion,args.n_class,device)
            # if cs_acc > best_cs_acc:
            #     best_cs_acc=cs_acc
            #     torch.save(model.state_dict(),os.path.join(save_path,"best{}_CS.pth".format(args.model_name)))

            print("Current accuracy:{:.3f},cross_subject accuracy:{:.3f}\nBest accuracy:{:.3f},Best cross_subject accuracy:{:.3f}"
                    .format(acc,cs_acc,best_acc,best_cs_acc))
            print("#-----------------------------------------------------------------------#")

        return [best_acc,best_cs_acc]

    elif args.mode == "inference":
        print("inference start")
        model.load_state_dict(torch.load("./models/best{}.pth".format(args.model_name)))

        if args.model_name=="GCN":
            cm,acc,t_mean=utils.test(0,model,test_loader,criterion,args.n_class,device,adj)
        else:
            cm,acc,t_mean=utils.test(0,model,test_loader,criterion,args.n_class,device)
        
        print("Current accuracy:{:.3f}".format(acc))
        return cm,acc,t_mean



if __name__ == "__main__":
    # Normal Function
    args=build_args("GCN")
    args.mode = "train"
    main(args)
    args.mode = "inference"
    cm,acc,t_mean=main(args)
    title="Confision matrix of "+args.model_name
    utils.v_confusion_matrix(cm,args.part_actions,title=title,save_path="./figure/{}_CM.png".format(args.model_name))
    utils.v_confusion_matrix(cm,args.part_actions,title=title,save_path="./figure/{}_CM.pdf".format(args.model_name))
    plt.show()

    #Channel Split Analysis
    # args=build_args("CNN")
    # utils.make_dir("./output")
    # n=2
    # with open ('./output/channel_result_{}.json'.format(n)) as json_file:
    #     result=json.load(json_file)
    # print(result)
    # channel_list=utils.select_channel(n,list(range(16)))
    # print(channel_list)
    # for channel in channel_list:
    #     args.channels=channel
    #     name="CH"+"_".join(str(c) for c in channel)
    #     print(name)

    #     args.mode="train"
    #     ACC=main(args)
    #     print(ACC)

    #     result[name]=ACC

    #     json_str = json.dumps(result, indent=4)
    #     with open('./output/channel_result_{}.json'.format(n), 'w') as json_file:
    #         json_file.write(json_str)

    #Model performance 
    # model_name="LDA"
    # for model_name in ["LDA","ANN","CNN","LSTM"]:
    #     print(model_name)
    #     args=build_args(model_name)
    #     CM=np.zeros((5,args.n_class,args.n_class))
    #     ACC=np.zeros(5)
    #     for i in range(5):
    #         args.subindex=i
    #         if model_name == "LDA":
    #             cm,acc=LDAmodel(args)
    #         else:
    #             args.mode = "train"
    #             main(args)
    #             args.mode = "inference"
    #             cm,acc=main(args)

    #         CM[i,:,:]=cm
    #         ACC[i]=acc

    #     save_dir="./output/model_performance"
    #     utils.make_dir(save_dir)
    #     np.save(save_dir+"/{}_CM.npy".format(model_name),CM)
    #     np.save(save_dir+"/{}_ACC.npy".format(model_name),ACC)
    #     # CM=np.rint(CM.mean(axis=0)) #rint取整
    #     # print(CM)
    #     # title="Confusion matrix of "+args.model_name
    #     # utils.v_confusion_matrix(cm,args.part_actions,title=title,save_path="./figure/{}_CM.png".format(args.model_name))
    #     # utils.v_confusion_matrix(cm,args.part_actions,title=title,save_path="./figure/{}_CM.pdf".format(args.model_name))
    #     # plt.show()

    
    #Time_delay Analysis
    # import time
    # for model_name in ["LDA","ANN","CNN","LSTM"]:
    #     print(model_name)
    #     args=build_args(model_name)
    #     Time=np.zeros(5)
    #     for i in range(5):
    #         args.subindex=i
    #         if model_name == "LDA":
    #             _,_,t_mean=LDAmodel(args)
    #         else:
    #             args.mode = "inference"
    #             _,_,t_mean=main(args)
    #         Time[i]=t_mean
    #     save_dir="./output/model_usedtime"
    #     utils.make_dir(save_dir)
    #     np.save(save_dir+"/{}_Time.npy".format(model_name),Time)
    #     print(Time)

    

