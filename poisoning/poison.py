import os
from sys import argv
from my_args import setup_argparse
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy.linalg as la

from gd_poisoners import *
from inits import *

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

#open and read the dataset file
def open_dataset(f,visualize):
	x,y = read_dataset_file(f)
	return np.matrix(x), y

#used by open_dataset()
def read_dataset_file(f):
  with open(f) as dataset:
    x = []
    y = []
    cols = dataset.readline().split(',')
    #print(cols)
    
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

#produce the sample data from the dataset
def sample_dataset(x, y, trnct, poisct, tstct, vldct, seed):
	size = x.shape[0]
	print('x_size:',size)

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

	for i in range(len(testy)):
	    testfile.write(','.join([str(val) for val in [testy[i]]+testx[i].tolist()[0]]) + '\n')
	testfile.close()

	for i in range(len(validy)):
	    validfile.write(','.join([str(val) for val in [validy[i]]+validx[i].tolist()[0]]) + '\n')
	validfile.close()

	for i in range(len(trainy)):
	    trainfile.write(','.join([str(val) for val in [trainy[i]]+trainx[i].tolist()[0]]) + '\n')
	#log the sample data in the file

	print('the rank of trainx:',la.matrix_rank(trainx))#calculate the rank of the matrix
	print('the shape of trainx',trainx.shape)

	totprop = args.poisct/(args.poisct + args.trainct)
	print('totprop:',totprop)

	timestart,timeend = None,None
	types = {'linreg': LinRegGDPoisoner,\
	         'lasso': LassoGDPoisoner,\
	         'enet': ENetGDPoisoner,\
	         'ridge': RidgeGDPoisoner}

	inits = {'levflip': levflip,\
	         'cookflip': cookflip,\
	         'alfatilt': alfa_tilt,\
	         'inflip': inf_flip,\
	         'ffirst': farthestfirst,\
	         'adaptive': adaptive,\
	         'randflip': randflip,\
	         'randflipnobd': randflipnobd,\
	         'rmml': rmml}

	bestpoisx, bestpoisy, besterr = None, None, -1
	init = inits[args.initialization]

	genpoiser = types[args.model](trainx, trainy, testx, testy, validx, validy,
                                  args.eta, args.beta, args.sigma, args.epsilon,
                                  args.multiproc,
                                  trainfile,resfile,args.objective,args.optimizey, colmap)
    #define the generater of the poisoning

	for initit in range(args.numinit): #numinit: number of times to attempt initialization
		poisx,poisy = init(trainx,trainy,int(args.trainct*totprop/(1-totprop)+0.5),genpoiser)
		#get the init poison points
		clf, _ = genpoiser.learn_model(np.concatenate((trainx,poisx),axis=0),trainy+poisy,None)
		#numpy.concatenate() is used to splice two arrays to one.
		#clf is the model which has fitted the data
		err = genpoiser.computeError(clf)[0]#err is the validation error
		print("Validation Error:", err)
		if err > besterr:
			bestpoisx, bestpoisy, besterr = np.copy(poisx), poisy[:], err
	poisx, poisy = np.matrix(bestpoisx), bestpoisy # the poison points which have been initted
	poiser = types[args.model](trainx, trainy, testx, testy, validx, validy,\
                               args.eta, args.beta, args.sigma, args.epsilon,\
                               args.multiproc, trainfile, resfile,\
                               args.objective, args.optimizey, colmap)

	for i in range(args.partct + 1):
		curprop = (i + 1)*totprop/(args.partct + 1)
		numsamples = int(0.5 + args.trainct*(curprop/(1 - curprop)))
		curpoisx = poisx[:numsamples,:]
		curpoisy = poisy[:numsamples]
		trainfile.write("\n")

		timestart = datetime.datetime.now()# log the start time
		poisres, poisresy = poiser.poison_data(curpoisx, curpoisy, timestart, args.visualize, newlogdir)
		#it seems that the poison_data function is used to produce the best poison data to max the MSE
		print(poisres.shape,trainx.shape)
		poisedx = np.concatenate((trainx,poisres),axis = 0)
		poisedy = trainy + poisresy
		#poisedx and poisedy are the raw data + poisoned data

		clfp, _ = poiser.learn_model(poisedx,poisedy,None)# the model has been poisoned
		clf = poiser.initclf                              # the model has not been poisoned, which is set at the init() of class poisoner
		if args.rounding:# round the x and y (four lose and five in ^_^)
			roundx,roundy = roundpois(poisres,poisresy)
			rpoisedx,rpoisedy = np.concatenate((trainx,roundx),axis = 0),trainy + roundy
			clfr, _ = poiser.learn_model(rpoisedx,rpoisedy,None)
			rounderr = poiser.computeError(clfr)

		errgrd = poiser.computeError(clf)
		err = poiser.computeError(clfp)

		timeend = datetime.datetime.now()

		towrite = [numsamples,-1,None,None,err[0],err[1],(timeend-timestart).total_seconds()]
		resfile.write(','.join([str(val) for val in towrite])+"\n")
		trainfile.write("\n")
		for j in range(numsamples):
			trainfile.write(','.join([str(val) for val in [poisresy[j]]+poisres[j].tolist()[0]])+'\n')
		if args.rounding:
			towrite = [numsamples,'r',None,None,rounderr[0],rounderr[1],(timeend-timestart).total_seconds()]
			resfile.write(','.join([str(val) for val in towrite])+"\n")
			trainfile.write("\nround\n")
			for j in range(numsamples):
				trainfile.write(','.join([str(val) for val in [roundy[j]]+roundx[j].tolist()[0]])+'\n')

		resfile.flush()
		trainfile.flush()
		os.fsync(resfile.fileno())
		os.fsync(trainfile.fileno())
   
	trainfile.close()
	testfile.close()

	print('-----------------')
	print("Unpoisoned:")
	print("Validation MSE:",errgrd[0])
	print("Test MSE:",errgrd[1])
	print('-----------------')
	print('Poisoned:')
	print("Validation MSE:",err[0])
	print("Test MSE:",err[1])
	print('-----------------')
	if args.rounding:
		print('-----------------')
		print("Rounded:")
		print("Validation MSE",rounderr[0])
		print("Test MSE:", rounderr[1])
		print('-----------------')

	print('end!')


if __name__=='__main__':
	print("start!\n")
	parser = setup_argparse()
	args = parser.parse_args()

	print("-----------------------------------------------------------")
	print(args)
	print("-----------------------------------------------------------")
	main(args)