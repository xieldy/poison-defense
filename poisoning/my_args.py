
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

    # counts
    parser.add_argument("-r", "--trainct",default=300, type=int,\
                            help='number of points to train models with')
    parser.add_argument("-t", "--testct",default=500, type=int,\
                            help = 'number of points to test models on')
    parser.add_argument("-v","--validct",default=250, type=int,\
                            help='size of validation set')
    parser.add_argument("-p", "--poisct",default=75, type=int,\
                            help='number of poisoning points')
    parser.add_argument("-s", "--partct",default=4, type=int,\
                            help='number of increments to poison with') 

    # seed for randmization
    parser.add_argument('-seed',type=int,help='random seed')

    #init strategy 
    parser.add_argument('-init','--initialization',default='inflip',
                            choices=['levflip', 'cookflip', 'alfatilt', 'randflip',\
                                     'randflipnobd', 'inflip', 'ffirst', 'adaptive',\
                                     'adaptilt','rmml'],\
                            help="init strategy")
    parser.add_argument('-numinit', type=int, default=1, help='number of times to attempt initialization')

    # params for gd poisoning
    parser.add_argument("-l", "--lambd",default=1, type=float,help='lambda value to use in poisoning;icml 2015')
    parser.add_argument("-n", "--epsilon",default=1e-3, type=float,help='termination condition epsilon;icml 2015')
    parser.add_argument("-a", "--eta",default=0.01, type=float,help='line search multiplier eta; icml 2015')
    parser.add_argument("-b", "--beta",default=0.05, type=float,help='line search decay beta; icml 2015')
    parser.add_argument("-i", "--sigma",default=0.9, type=float,help='line search termination lowercase sigma;icml 2015')

    # enable multi processing
    parser.add_argument("-mp","--multiproc",action='store_true',\
                            help='enable to allow for multiprocessing support')

    # objective
    parser.add_argument("-obj","--objective",default=0,type=int,
                            help="objective to use (0 for train, 1 for validation, 2 for norm difference)")
    parser.add_argument('-opty','--optimizey',action='store_true',
                            help='optimize the y values of poisoning as well')

    # round 
    parser.add_argument("-rnd",'--rounding',action='store_true',help='to round or not to round')


    return parser
   
