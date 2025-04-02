import sys
import argparse
from config import load_config
from main_with_e import main

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ## Parameter setting
    current_dir = sys.path[0]
    parser.add_argument("--path", type=str, default=current_dir)
    parser.add_argument("--data_path", type=str, default='/data/', help="Path of datasets.")
    parser.add_argument("--save_file", type=str, default="res_para.txt", help="Path of datasets.")
    parser.add_argument("--device", type=str, default="0", help="Device: cuda:num or cpu")
    parser.add_argument("--use_seed", action='store_true', default=True)
    parser.add_argument("--seed", type=int, default=40, help="Random seed, default is 42.")
    parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument("--norm", action='store_true', default=True, help="Normalize the feature.")
    parser.add_argument("--block", type=int, default="2", help="block")  # network layers
    parser.add_argument("--thre1", type=float, default="0.01", help="thre1")  # parameter for L1 regularization
    parser.add_argument("--thre2", type=float, default="0.001", help="thre2")  # parameter for L21 regularization
    parser.add_argument("--lr", type=float, default="0.05", help="lr") # Learning rate
    parser.add_argument("--epoch", type=int, default="100", help="epochs")
    args =parser.parse_args()

    dataset = {1: "animals", 2: "cifar10", 3: "flickr", 4: "iaprtc12"
               , 5: "NUSWIDEOBJ",  6: "YTF50", 7: "YTF100", 8: "Caltech101all"}  # dataset list

    select_dataset = [3]
    # select_dataset = list(range(1, 9))
    for i in select_dataset:
        main(dataset[i], args)

