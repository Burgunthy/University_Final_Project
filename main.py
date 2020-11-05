
## 라이브러리 추가하기
import argparse

from train_b import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="Regression Tasks such as",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="test", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--code_size", default=128, type=int, dest="code_size")
parser.add_argument("--init_size", default=4, type=int, dest="init_size")
parser.add_argument("--lr", default=0.001, type=float, dest="lr")
parser.add_argument("--batch_size", default=1, type=int, dest="batch_size")
parser.add_argument("--num_iteration", default=1000, type=int, dest="num_iteration")
parser.add_argument("--n_critic", default=1, type=int, dest="n_critic")

parser.add_argument("--data_dir", default="./drive/My Drive/pytorch/final_project/datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./drive/My Drive/pytorch/final_project/checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./drive/My Drive/pytorch/final_project/log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./drive/My Drive/pytorch/final_project/result", type=str, dest="result_dir")

parser.add_argument("--wgt", default=1e2, type=float, dest="wgt")

args = parser.parse_args()

if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)


















