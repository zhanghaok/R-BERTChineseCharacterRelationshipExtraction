import os

from hparams import hparams
from train import train
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def trainer():
    train(hparams)



def predicter():
    pass


if __name__ == '__main__':
    trainer()