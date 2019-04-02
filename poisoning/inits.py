import numpy as np
import bisect

# ------------------------------------------------------------------------------- 
def levflip(x,y,count,poiser):
  allpoisy = []
  clf,_ = poiser.learn_model(x,y,None)
  mean = np.ravel(x.mean(axis=0))#.reshape(1,-1)
  corr = np.dot(x.T,x) + 0.01*np.eye(x.shape[1])
  invmat = np.linalg.pinv(corr)
  hmat = x*invmat*np.transpose(x)

  alllevs = [hmat[i,i] for i in range(x.shape[0])]
  totalprob = sum(alllevs)
  allprobs = [0]
  for i in range(len(alllevs)):
    allprobs.append(allprobs[-1]+alllevs[i])
  allprobs=allprobs[1:]
  poisinds = []
  for i in range(count):
    a = np.random.uniform(low=0,high=totalprob)
    curind = bisect.bisect_left(allprobs,a)
    poisinds.append(curind)
    if clf.predict(x[curind].reshape(1,-1))<0.5:
      allpoisy.append(1)
    else:
      allpoisy.append(0)

  return x[poisinds],allpoisy

# ------------------------------------------------------------------------------- 
def cookflip(x,y,count,poiser):
  allpoisy = []
  clf,_ = poiser.learn_model(x,y,None)
  preds = [clf.predict(x[i].reshape(1,-1)) for i in range(x.shape[0])]
  errs = [(y[i]-preds[i])**2 for i in range(x.shape[0])]
  mean = np.ravel(x.mean(axis=0))#.reshape(1,-1)
  corr = np.dot(x.T,x) + 0.01*np.eye(x.shape[1])
  invmat = np.linalg.pinv(corr)
  hmat = x*invmat*np.transpose(x)
  
  allcooks = [hmat[i,i]*errs[i]/(1-hmat[i,i])**2 for i in range(x.shape[0])]

  totalprob = sum(allcooks)

  allprobs = [0]
  for i in range(len(allcooks)):
    allprobs.append(allprobs[-1]+allcooks[i])
  allprobs=allprobs[1:]
  poisinds = []
  for i in range(count):
    a = np.random.uniform(low=0,high=totalprob)
    curind = bisect.bisect_left(allprobs,a)
    poisinds.append(curind)
    if clf.predict(x[curind].reshape(1,-1))<0.5:
      allpoisy.append(1)
    else:
      allpoisy.append(0)

  return x[poisinds],allpoisy

# ------------------------------------------------------------------------------- 
def alfa_tilt(X_tr, Y_tr,count,poiser):
  inv_cov = (0.01 * np.eye(X_tr.shape[1]) + np.dot(X_tr.T, X_tr)) ** -1
  H = np.dot(np.dot(X_tr, inv_cov), X_tr.T)
  randplane = np.random.standard_normal(size=X_tr.shape[1]+1)
  w, b = randplane[:-1], randplane[-1]
  preds = np.dot(X_tr,w)+b
  yvals = preds.clip(0,1)
  yvals = 1-np.floor(0.5+yvals)
  diff = yvals - Y_tr
  print(diff)
  yvals = yvals.tolist()[0]
  changes = np.dot(diff,H).tolist()[0]
  changes = [max(a,0) for a in changes]
 
  totalprob = sum(changes)
  allprobs = [0]
  poisinds = []
  for i in range(X_tr.shape[0]):
    allprobs.append(allprobs[-1]+changes[i])
  allprobs = allprobs[1:]
  for i in range(count):
    a = np.random.uniform(low=0,high=totalprob)
    poisinds.append(bisect.bisect_left(allprobs,a))
  return X_tr[poisinds], [yvals[a] for a in poisinds]
# ------------------------------------------------------------------------------- 
def alfatilt(x,y,count,poiser):
  trueclf,_ = poiser.learn_model(x,y,None)
  truepreds = trueclf.predict(x)

  goalmodel = np.random.uniform(low=-1/sqrt(x.shape[1]),high=1/sqrt(x.shape[1]),shape=(x.shape[1]+1))
  goalpreds = np.dot(x,goalmodel[:-1])+goalmodel[-1].item()

  svals = np.square(trueclf.predict(x)-y)  # squared error
  svals = svals/svals.max()
  qvals = np.square(goalpreds-y)
  qvals = qvals/qvals.max()

  flipscores = (svals+qvals).tolist()

  totalprob = sum(flipscores)
  allprobs = [0]
  allpoisy = []
  for i in range(len(flipscores)):
    allprobs.append(allprobs[-1]+flipscores[i])
  allprobs=allprobs[1:]
  poisinds = []
  for i in range(count):
    a = np.random.uniform(low=0,high=totalprob)
    poisinds.append(bisect.bisect_left(allprobs,a))
    if truepreds[curind]<0.5:
      allpoisy.append(1)
    else:
      allpoisy.append(0)

  return x[poisinds],allpoisy

# ------------------------------------------------------------------------------- 
def infflip(x,y,count,poiser):
  mean = np.ravel(x.mean(axis=0))#.reshape(1,-1)
  corr = np.dot(x.T,x) + 0.01*np.eye(x.shape[1])
  invmat = np.linalg.pinv(corr)
  hmat = x*invmat*np.transpose(x)
  allgains = []
  for i in range(x.shape[0]):
    posgain = (np.sum(hmat[i])*(1-y[i]),1)
    neggain = (np.sum(hmat[i])*y[i],0)
    allgains.append(max(posgain,neggain))

  totalprob = sum([a[0] for a in allgains])
  allprobs = [0]
  for i in range(len(allgains)):
    allprobs.append(allprobs[-1]+allgains[i][0])
  allprobs=allprobs[1:]
  poisinds = []
  for i in range(count):
    a = np.random.uniform(low=0,high=totalprob)
    poisinds.append(bisect.bisect_left(allprobs,a))
  gainsy = [allgains[ind][1] for ind in poisinds]

  #sortedgains = sorted(enumerate(allgains),key = lambda tup: tup[1])[:count]
  #poisinds = [a[0] for a in sortedgains]
  #bestgains = [a[1][1] for a in sortedgains]

  return x[poisinds], gainsy
# ------------------------------------------------------------------------------- 
def inf_flip(X_tr, Y_tr,count,poiser):
  Y_tr = np.array(Y_tr)
  inv_cov = (0.01 * np.eye(X_tr.shape[1]) + np.dot(X_tr.T, X_tr)) ** -1
  H = np.dot(np.dot(X_tr, inv_cov), X_tr.T)
  bests = np.sum(H,axis = 1)
  room = .5 + np.abs(Y_tr-0.5)
  yvals = 1-np.floor(0.5+Y_tr)
  stat = np.multiply(bests.ravel(),room.ravel())
  stat = stat.tolist()[0]
  totalprob = sum(stat)
  allprobs = [0]
  poisinds = []
  for i in range(X_tr.shape[0]):
    allprobs.append(allprobs[-1]+stat[i])
  allprobs = allprobs[1:]
  for i in range(count):
    a = np.random.uniform(low=0,high=totalprob)
    poisinds.append(bisect.bisect_left(allprobs,a))
  
  return X_tr[poisinds], [yvals[a] for a in poisinds]




# -------------------------------------------------------------------------------   
def farthestfirst(x,y,count,poiser):
  allpoisy = []
  clf,_ = poiser.learn_model(x,y,None)
  preds = [clf.predict(x[i].reshape(1,-1)) for i in range(x.shape[0])]
  errs = [(y[i]-preds[i])**2 for i in range(x.shape[0])]
  totalprob = sum(errs)
  allprobs = [0]
  for i in range(len(errs)):
    allprobs.append(allprobs[-1]+errs[i])
  allprobs=allprobs[1:]
  poisinds = []
  for i in range(count):
    a = np.random.uniform(low=0,high=totalprob)
    curind = bisect.bisect_left(allprobs,a)
    poisinds.append(curind)
    if preds[curind]<0.5:
      allpoisy.append(1)
    else:
      allpoisy.append(0)

  return x[poisinds],allpoisy
# ------------------------------------------------------------------------------- 
def adaptive(X_tr, Y_tr, count,poiser):
  Y_tr_copy = np.array(Y_tr)
  X_tr_copy = np.copy(X_tr)
  print(np.allclose(X_tr_copy,X_tr))
  room = .5+np.abs(Y_tr_copy)
  yvals = 1-np.floor(0.5+Y_tr_copy)
  diff = (yvals-Y_tr_copy).ravel()
  poisinds = []
  X_pois = np.zeros((count,X_tr.shape[1]))
  Y_pois = []
  for i in range(count):
    print(X_tr_copy.shape,diff.shape)
    inv_cov = np.linalg.inv(0.01 * np.eye(X_tr_copy.shape[1]) + np.dot(X_tr_copy.T, X_tr_copy))
    H = np.dot(np.dot(X_tr_copy, inv_cov), X_tr_copy.T)
    bests = np.sum(H,axis=1)
    stat = np.multiply(bests.ravel(),diff)
    #indtoadd = np.argmax(stat)
    indtoadd = np.random.choice(stat.shape[0],p=np.abs(stat)/np.sum(np.abs(stat)))
    print(indtoadd)
    X_pois[i] = X_tr_copy[indtoadd,:]
    X_tr_copy = np.delete(X_tr_copy,indtoadd,axis=0)
    diff = np.delete(diff,indtoadd,axis=0)
    Y_pois.append(yvals[indtoadd])
    yvals = np.delete(yvals,indtoadd,axis=0)
  print(X_pois)
  print(Y_pois)
  return np.matrix(X_pois),Y_pois

# ------------------------------------------------------------------------------- 
def randflip(X_tr, Y_tr, count,poiser):
  poisinds = np.random.choice(X_tr.shape[0],count,replace=False)
  print("Points selected: ",poisinds)
  #Y_pois = [1-Y_tr[i] for i in poisinds]  # this is for validating yopt, not for initialization
  Y_pois = [1 if 1-Y_tr[i]>0.5 else 0 for i in poisinds]  # this is the flip all the way implementation
  return np.matrix(X_tr[poisinds]), Y_pois
# ------------------------------------------------------------------------------- 
def randflipnobd(X_tr, Y_tr, count,poiser):
  poisinds = np.random.choice(X_tr.shape[0],count,replace=False)
  print("Points selected: ",poisinds)
  Y_pois = [1-Y_tr[i] for i in poisinds]  # this is for validating yopt, not for initialization
  #Y_pois = [1 if 1-Y_tr[i]>0.5 else 0 for i in poisinds]  # this is the flip all the way implementation
  return np.matrix(X_tr[poisinds]), Y_pois
# -------------------------------------------------------------------------------
def rmml(X_tr,Y_tr, count,poiser):
  print(X_tr.shape, len(Y_tr), count)
  mean = np.ravel(X_tr.mean(axis=0))#.reshape(1,-1)
  covar = np.dot((X_tr-mean).T,(X_tr-mean))/X_tr.shape[0] + 0.01*np.eye(X_tr.shape[1])
  model = linear_model.Ridge(alpha=.01)
  model.fit(X_tr,Y_tr)
  allpoisx = np.random.multivariate_normal(mean,covar,size=count)
  allpoisx[allpoisx>=0.5] = 1
  allpoisx[allpoisx<0.5] = 0
  poisy = model.predict(allpoisx)
  poisy = 1- poisy
  poisy[poisy>=0.5] = 1
  poisy[poisy<0.5] = 0
  print(allpoisx.shape,poisy.shape)
  for i in range(count):
    curpoisxelem = allpoisx[i,:]
    for col in colmap:
      vals = [(curpoisxelem[j],j) for j in colmap[col]]
      topval,topcol = max(vals)
      for j in colmap[col]:
        if j!=topcol:
          curpoisxelem[j]=0
      if topval>1/(1+len(colmap[col])):
        curpoisxelem[topcol]=1
      else:
        curpoisxelem[topcol]=0
    allpoisx[i]=curpoisxelem
  return np.matrix(allpoisx),poisy.tolist()


