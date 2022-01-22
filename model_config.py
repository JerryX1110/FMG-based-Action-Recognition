import argparse
#dataset1
_CLASS_NAME = {"dataset1":["Relaxed_Hand",
                    "Force1_Still","Force1_Motion","Force2_Still","Force2_Motion",
                    "Force3_Still","Force3_Motion","Force4_Still","Force4_Motion",
                    "Finger_Abduction","Clench","V_Aign","Clench_Abduction",
                    "Wrist_Swing","Rotating_Arm","Force1_Rotation"],

                "dataset2":["Keep_Still","Screw_with_Screwdriver","Lift_a_Bottle","Screw_the_bottle_Cap",
                "use_Glue_Gun_Still","use_Glue_Gun","Lift_up_Object","KeyBoarding","Cut_the_Rope",
                "Tie_a_Rope","Rub_a_Surface_with_sandpaper","WriTing","Wipe_the_Table"]
}

_CLASS_NAME_SHORT = {"dataset1":["RH",
                    "F1S","F1M","F2S","F2M",
                    "F3S","F3M","F4S","F4M",
                    "FA","CL","VA","CA",
                    "WS","RA","F1R"],

                    "dataset2":["KS","SS","LB","SC",
                    "GGS","GG","LO","KB","CR",
                    "TR","RS","WR","WT"]
}


_MODEL_HYPER_PARAMS = {
    "LSTM":{
        "epochs":30,
        "batch_size":4,
        "lr":0.01,
        "hidden_dim":32,
        "n_layer":2   
    },
    "ANN":{
        "epochs":4,
        "batch_size":4,
        "lr":0.01,
        "hidden_dim":32,
        "n_layer":2   
    },
    "CNN":{
        "epochs":5,
        "batch_size":4,
        "lr":0.01,
        "hidden_dim":32,
        "n_layer":2   
    },
    "GCN":{
        "epochs":15,
        "batch_size":1,
        "lr":0.01,
        "hidden_dim":16,
        "n_layer":2   
    }

}


def build_args(model_name=None):
    parser = argparse.ArgumentParser("This script is used for the FMG-based Action Classification.")

    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--model_name", default="LSTM", type=str)
    parser.add_argument("--model_file", default=None, type=str)
    parser.add_argument("--dataset", default="dataset1", type=str)
    args = parser.parse_args()

    # args.class_name=_CLASS_NAME
    args.class_name=_CLASS_NAME_SHORT[args.dataset]
    args.class_dict={action_cls:idx for idx, action_cls in enumerate(args.class_name)}
    if args.dataset=="dataset1":
        args.data_root="D://数据集/window_data"
        args.repeat=2
        args.part_actions=args.class_name[0:1]+args.class_name[2:3]+args.class_name[9:] #choose part of actions
    elif args.dataset=="dataset2":
        args.data_root="D://数据集/window_data_new"
        args.repeat=3
        args.part_actions=args.class_name[0:4]+args.class_name[5:]
    args.n_class=len(args.part_actions)
    print("number of class:{}\naction list:{}".format(args.n_class,args.part_actions))
    
    #prepare data
    args.L_win=100
    args.stride=5
    args.channels=list(range(16))   #choose part of channels
    print("length of window:{}\nstride of window:{}".format(args.L_win,args.stride))
    
    #train_test_subject
    args.subindex=[0]
    args.test_ratio=0.2
    print("subject{}'s data".format(args.subindex))


    if model_name is not None:
        args.model_name = model_name

    if model_name != "LDA":
        args.epochs=_MODEL_HYPER_PARAMS[args.model_name]["epochs"]
        args.batch_size=_MODEL_HYPER_PARAMS[args.model_name]["batch_size"]
        args.lr=_MODEL_HYPER_PARAMS[args.model_name]["lr"]
        args.hidden_dim=_MODEL_HYPER_PARAMS[args.model_name]["hidden_dim"]
        args.n_layer=_MODEL_HYPER_PARAMS[args.model_name]["n_layer"]

    return args

if __name__ == "__main__":
    build_args("LSTM")

