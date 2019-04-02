import os
from sys import argv
from my_args import setup_argparse
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


#open these files and log the args(including defult args)
def open_logging_files(logdir,modeltype,logind,args):
	myname = str(modeltype)+str(logind)
	logdir = logdir + os.path.sep + myname
	if not os.path.exists(logdir):
		os.makedirs(logdir)
	with open(os.path.join(logdir,'cmd'),'w') as cmdfile:
		cmdfile.write(' '.join(['python3'] + argv))
		cmdfile.write('\n')
		for arg in args.__dict__:
			cmdfile.write('{}: {}\n'.format(arg,args.__dict__[arg]))

	trainfile = open(logdir + os.path.sep + "train.txt",'w')
	testfile = open(logdir + os.path.sep + "test.txt",'w')
	validfile = open(logdir + os.path.sep + "valid.txt",'w')
	resfile = open(logdir + os.path.sep + "err.txt",'w')
	resfile.write('poisct,itnum,obj_diff,obj_val,val_mse,test_mse,time\n')
	return trainfile,testfile,validfile,resfile,logdir

def open_dataset(f,visualize):
	x,y = read_dataset_file(f)
	return np.matrix(x), y

def read_dataset_file(f):
  with open(f) as dataset:
    x = []
    y = []
    cols = dataset.readline().split(',')
    print(cols)
    
    global colmap
    colmap = {}
    for i, col in enumerate(cols):
      if ':' in col:
        if col.split(':')[0] in colmap:
          colmap[col.split(':')[0]].append(i-1)
        else:
          colmap[col.split(':')[0]] = [i-1]
    for line in dataset:
      line = [float(val) for val in line.split(',')]
      y.append(line[0])
      x.append(line[1:])

    return np.matrix(x), y


def sample_dataset(x, y, trnct, poisct, tstct, vldct, seed):
	size = x.shape[0]
	print(size)

	np.random.seed(seed)
	fullperm = np.random.permutation(size)
	#if input a matrix, the function will return a shuffle of the matrix;
	#if input a number, the function will return a shuffle of the arange(from 1 to size and shuffle them);

	sampletrn = fullperm[:trnct]
	sampletst = fullperm[trnct:trnct + tstct]
	samplevld = fullperm[trnct + tstct:trnct + tstct + vldct]
	#use the shuffle arange to choose the train set, test set and validaton set randomly.
	samplepois = np.random.choice(size, poisct)
	#the function is to produce a poisct-size data set from a size-size random arange.

	trnx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in sampletrn])
	trny = [y[row] for row in sampletrn]


	tstx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in sampletst])
	tsty = [y[row] for row in sampletst]

	poisx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in samplepois])
	poisy = [y[row] for row in samplepois]

	vldx = np.matrix([np.array(x[row]).reshape((x.shape[1],)) for row in samplevld])
	vldy = [y[row] for row in samplevld]
	#make up the data set with the sample list

	return trnx, trny, tstx, tsty, poisx, poisy, vldx, vldy


def main(args):
	trainfile, testfile, validfile, resfile, newlogdir =\
		open_logging_files(args.logdir, args.model, args.logind, args)#open these files
	x,y = open_dataset(args.dataset, args.visualize)
	trainx, trainy, testx, testy, poisx, poisy, validx, validy = \
        sample_dataset(x, y, args.trainct, args.poisct, args.testct, args.validct,\
                       args.seed)
    #produce the sample data from the dataset

	print('end!')


if __name__=='__main__':
	print("start!\n")
	parser = setup_argparse()
	args = parser.parse_args()

	print("-----------------------------------------------------------")
	print(args)
	print("-----------------------------------------------------------")
	main(args)