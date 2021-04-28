import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser("python train.py --gpu_nums --epochs")
    parser.add_argument("--gpu_nums", help="input gpu nums")
    parser.add_argument("--epochs", help="train epochs")
    args = parser.parse_args()
    print(args.epochs)