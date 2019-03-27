import os
from sys import argv
from my_args import setup_argparse


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
  if visualize:#Why not use the data in the file but produce random data when '-vis' is set?
    rng = np.random.RandomState(1)#set the random seed = 1
    random_state = 1
    x, y = make_regression(n_samples=300, n_features=1, random_state=random_state, noise=15.0, bias=1.5)
    #make_regression:a function in sklearn, which can random produce some data about linear_regression
   
    x = (x-x.min())/(x.max()-x.min())
    y = (y-y.min())/(y.max()-y.min())

    plt.plot(x, y, 'k.')
    global colmap
    colmap = []
  else:
    x,y = read_dataset_file(f)

  return np.matrix(x), y


def main(args):
	trainfile, testfile, validfile, resfile, newlogdir =\
		open_logging_files(args.logdir, args.model, args.logind, args)#open these files
	x,y = open_dataset(args.dataset, args.visualize)
	print('end!')


if __name__=='__main__':
	print("start!\n")
	parser = setup_argparse()
	args = parser.parse_args()

	print("-----------------------------------------------------------")
	print(args)
	print("-----------------------------------------------------------")
	main(args)