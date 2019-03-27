
import argparse

def setup_argparse():
    parser = argparse.ArgumentParser(description='handle script inputs')

    # dataset file
    parser.add_argument("-d", "--dataset",default='../datasets/house-processed.csv',\
                            help='dataset filename (includes path)')

    parser.add_argument("-ld", "--logdir",default="../results",\
                            help='directory to store output')

    parser.add_argument("-li","--logind",default=0,\
                            help='output files will be err{model}{outputind}.txt, train{model}{outputind}.txt, test{model}{outputind}.txt')

    parser.add_argument("-m", "--model",default='linreg',\
                            choices=['linreg','lasso','enet','ridge'],\
                            help="choose linreg for linear regression, lasso for lasso, enet for elastic net, or ridge for ridge")

    # visualize simple case
    parser.add_argument("-vis",'--visualize',action='store_true',help="visualize dataset")

    return parser
   
