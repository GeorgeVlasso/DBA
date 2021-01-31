import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from PIL import Image
import tensorflow as tf
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from tensorflow.keras.models import model_from_json
from sklearn.utils import check_random_state
from tensorflow.keras import optimizers
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from numpy.random import choice 
import matplotlib.patches as patches 
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cv2
from sklearn.model_selection import train_test_split
import os
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import random


my_parser = argparse.ArgumentParser()
my_parser.add_argument('-train_annotators', action='store_true')


args = my_parser.parse_args()

seed_value= 0
import random
random.seed(seed_value)


os.environ['PYTHONHASHSEED'] = str(seed_value)
tf.random.set_seed(100) 
rs = check_random_state(12345) 

print("Loading pretrained models and embedded data")

json_file = open('models/decode_clb.json', 'r')
loaded_decode_json = json_file.read()
json_file.close()
decoder = model_from_json(loaded_decode_json)
decoder.load_weights("models/decode_clb.h5")
print("Loaded decoder from disk")

json_file = open('models/cnn_clb3.json', 'r')
loaded_cnn_json = json_file.read()
json_file.close()
cnn = model_from_json(loaded_cnn_json)
cnn.load_weights("models/cnn_clb3.h5")
print("Loaded cnn from disk")

json_file = open('models/code_clb.json', 'r')
loaded_cnn_json = json_file.read()
json_file.close()
code = model_from_json(loaded_cnn_json)
code.load_weights("models/code_clb.h5")
print("Loaded encoder from disk")


embeddings = np.load("data/embs_00001-009000.npy")
preds = np.load("data/preds_00001-009000.npy")
predss = np.array(preds >= 0.5).astype("int32")

c_batch = np.load("data/celeba_batch.npy.npz")["arr_0"]

decoded = decoder.predict(code.predict(c_batch))
emb_batch = code.predict(c_batch)
true_preds = np.array([x[0] for x in cnn.predict(c_batch)])
fake_preds = np.array([x[0] for x in cnn.predict(decoder(code(c_batch)))] )
true_labs  = (true_preds > .5).astype("int32")
fake_labs  = (fake_preds > .5).astype("int32")



def Predict(x):
  dec = decoder.predict(x)
  p = (np.array([y[0] for y in cnn.predict(dec)]) >= 0.5).astype("int32")
  return(p)

def Predict_proba(x):
  dec = decoder.predict(x)
  p = np.array([y[0] for y in cnn.predict(dec)]) 
  return(p) 


print("Loaded everything")


path = "Figures"

wd = os.getcwd()

try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s " % path)

tf.random.set_seed(100) 
rs = check_random_state(1234)  




def find_db_index_k(x, y):
  k = np.where(np.abs(x-0.5) == np.abs(x-0.5).min())
  return x[k[0][0]],y[k[0][0]]

def find_db_index_k(x, y):
  k = np.where(np.abs(x-0.5) == np.abs(x-0.5).min())
  return x[k[0][0] -5 : k[0][0] + 5],y[k[0][0] -5 : k[0][0] + 5]

allAttributes = pd.read_csv('data/list_attr_celeba.csv.zip')
print(allAttributes)
Male = (allAttributes.loc[0:8999,"Male"] > 0).astype("int32")
Smiling = (allAttributes.loc[0:8999,"Smiling"] > 0).astype("int32")
Bangs = (allAttributes.loc[0:8999,"Bangs"] > 0).astype("int32")
Pale = (allAttributes.loc[0:8999,"Pale_Skin"] > 0).astype("int32")
Open_mouth = (allAttributes.loc[0:8999,"Mouth_Slightly_Open"] > 0).astype("int32")
Young = (allAttributes.loc[0:8999,"Young"] > 0).astype("int32")
Blond = (allAttributes.loc[0:8999,"Blond_Hair"] > 0).astype("int32")
Make_up = (allAttributes.loc[0:8999,"Heavy_Makeup"] > 0).astype("int32")
Blurry = (allAttributes.loc[0:8999,"Blurry"] > 0).astype("int32")

all_atts = np.vstack((Male, Smiling,  Bangs, Pale, Open_mouth, Young, Blond, Make_up, Blurry)).T
all_names = np.array(["Male", "Smiling",  "Bangs", "Pale", "Open_mouth","Young", "Blond", "Make_up", "Blurry"])


#Prepare data for annotators

seed_value= 0

random.seed(seed_value)

os.environ['PYTHONHASHSEED'] = str(seed_value)
tf.random.set_seed(100) 
rs = check_random_state(1234) 



X_train, X_test, y_train, y_test = train_test_split(embeddings, all_atts)

def train_annotator(x):
  lr1 = LogisticRegression(penalty  = 'l2', solver = "liblinear")
  lr1.fit(X_train, y_train[:,np.where(all_names == x)[0]])
  return(lr1)

def evaluate_annotator(a, x):
  return(a.score(X_test, y_test[:,np.where(all_names == x)[0]]))  

print("Approximate Prediction MAE loss: {} ".format(np.mean(np.abs(true_preds - fake_preds))))
print("Approximate Prediction 0/1 loss: {} ".format(np.mean(true_labs != fake_labs)))

#Reconstruction illustration

ids = [11,12,14]
fig = plt.figure(figsize=(5, 7))
outer = gridspec.GridSpec(1, 2, wspace=0, hspace=0)
s = [-1,0,1]
for i in range(2):
    inner = gridspec.GridSpecFromSubplotSpec(3, 1,
                    subplot_spec=outer[i], wspace=0, hspace=0)
    
    for j in range(3):
        ax = plt.Subplot(fig, inner[j])
        if i == 0:
          ax.imshow(c_batch[ids[j]])
          prr = cnn(c_batch[ids[j]:ids[j] + 1]).numpy()[0][0].round(4)
        else:
          ax.imshow((decoded[ids[j]:ids[j] + 1]).reshape((128,128,3))) 
          prr = Predict_proba(emb_batch[ids[j]:ids[j]+1])[0].round(4) 
        
        
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        
        
        ax.text(.52,.91,str(prr), fontsize = 15, c = "red",
        horizontalalignment='center',
        transform=ax.transAxes)  
        fig.add_subplot(ax)



fig.savefig(wd +"/"+ "Figures" + "/"  + "reconstruction_quality.png")
plt.clf()




def annotator(i):
  print(all_names[i])
  m = Sequential()
  m.add(Dense(1,  
                activation = 'sigmoid',
                kernel_regularizer = L1L2(l1 = 0.0, l2 = 0.1),
                input_dim = 100))  
  m.compile(optimizer = 'sgd',loss = 'binary_crossentropy', metrics = ['accuracy'])
  if vars(args)["train_annotators"]:
    filepath= "ano" + str(i) + ".best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    callbacks_list = [checkpoint]
    m.fit(X_train,y_train[:,i], epochs = 50 , validation_data= (X_test, y_test[:,i]), callbacks = callbacks_list)
  m.load_weights("models/"+"ano" + str(i) + ".best.hdf5")
  return m

ano_1 = Sequential()

annotators = [annotator(i) for i in range(9)]


def annotate(x):
  def P(a,x):
    return(np.squeeze(a.predict(x)))
  if x.shape[0] == 1:
    return(np.squeeze(np.vstack((P(ano_1,x), P(ano_2,x) , P(ano_3,x) ,\
     P(ano_4,x) , P(ano_5,x) , P(ano_6,x), P(ano_7,x), P(ano_8,x), P(ano_9,x) )).T.reshape((1,-1))))
  else:
    return(np.squeeze(np.vstack((P(ano_1,x), P(ano_2,x) , P(ano_3,x) ,\
     P(ano_4,x) , P(ano_5,x) , P(ano_6,x), P(ano_7,x), P(ano_8,x), P(ano_9,x) )).T))


ano_1 = annotators[0]
ano_2 = annotators[1]
ano_3 = annotators[2]
ano_4 = annotators[3]
ano_5 = annotators[4]
ano_6 = annotators[5]
ano_7 = annotators[6]
ano_8 = annotators[7]
ano_9 = annotators[8]


ano_coefs = [a.get_weights()[0].reshape((1,-1)) for a in annotators]

def n(x):
  return(x / np.sqrt(np.sum(x**2)))

fig = plt.figure(figsize=(10, 10))
outer = gridspec.GridSpec(1, 9, wspace=0, hspace=0)
s = [-3.5,0,3.5]
for i in range(3):
    inner = gridspec.GridSpecFromSubplotSpec(9, 1,
                    subplot_spec=outer[i], wspace=0, hspace=0)

    for j in range(9):
        ax = plt.Subplot(fig, inner[j])
        ax.imshow(decoder(embeddings[200:201] + s[i]*n(ano_coefs[j])).numpy().reshape((128,128,3)))
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 2:
          ax.set_title(str(all_names[j]), x = 2, y = 0.4, fontsize = 13, fontweight = 150)
          
        fig.add_subplot(ax)

fig.savefig(wd +"/"+ "Figures" + "/"  + "attribute_vectors_celeba.png")
plt.clf()


mean_emb = embeddings.mean(axis = 0).reshape((1,-1))
scale_emb = np.std(embeddings, axis= 0).reshape((1,-1))

def walk_db(co, step):
  return(step*n(co[0]*n(ano_coefs[0]) +
                co[1]*n(ano_coefs[1]) +
                co[2]*n(ano_coefs[2]) +
                co[3]*n(ano_coefs[3]) +
               
                co[4]*n(ano_coefs[4]) + 
                co[5]*n(ano_coefs[5]) +
                co[6]*n(ano_coefs[6]) +
                co[7]*n(ano_coefs[7]) +
                co[8]*n(ano_coefs[8])  
                ).reshape((1,-1)))


#Run DBA-ATT and LIME-ATT             

ds = []
coefss4 = np.zeros(9)
coefssl = np.zeros(9)
coefss4l2 = np.zeros(9)
faiths  = []
faithg = []
balances = []
balance_l = []
paths3_o = np.zeros(100)
paths3_g = np.zeros(100)
pred     = Predict(embeddings)
mean_emb = embeddings.mean(axis = 0).reshape((1,-1))
scale_emb = np.std(embeddings, axis= 0).reshape((1,-1))

def find_db_index_k(x, y):
  k = np.where(np.abs(x-0.5) == np.abs(x-0.5).min())
  return x[k[0][0]],y[k[0][0]]

def _dists(a, b):
   return np.sqrt(np.sum((a - b) ** 2, axis = tuple(range(1, len(embeddings.shape)))))




correct = np.where(fake_labs == true_labs)[0]  #Get points for which label is stable
to_explain = correct[0:30] 

for idc in to_explain:
  
  difdb1 = 10
  K = 0
  print("Example" + str(idc))
  examples = code.predict(c_batch[idc:idc+1]).reshape((1,-1))
  print(annotate(examples).round())
  an_ex    = annotate(examples)
  ex_pred = Predict(examples)
  print("Probability: " + str(Predict_proba(examples)))
  # Select set of rivals

  rivs = embeddings[pred != ex_pred][0:1000]
  rivs_ord = rivs[np.argsort(_dists(rivs,examples))]
  
  a = np.copy(rivs_ord)
  b = np.repeat(examples, a.shape[0], axis=0)

  mean_dist = 100000
  
  while mean_dist > 0.00001:

      mid = a/2 + b/2
      mid_pred = Predict((mid))
      same_class_as_example = mid_pred == ex_pred

      a[~same_class_as_example] = mid[~same_class_as_example]
      b[same_class_as_example] = mid[same_class_as_example]

      mean_dist = np.max(_dists(a, b))
      


  dists = _dists(a,examples)

  best = np.argmin(dists)
  z = a[best]
  d = dists[best]
  
  
  # Sample 

  f = 1 

  def create_negative_ver(cent , i, step):
    return(cent - step*(n(ano_coefs[i]) 
                ).reshape((1,-1)))

  def create_positive_ver(cent, i,step):
    return( cent + step*(n(ano_coefs[i]) 
                ).reshape((1,-1)))       

  fs = np.array([0.1,0.2,0.3,.4,0.5,0.6,0.7,0.8,0.9,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,9,9.5,10])

  
  samp =np.zeros(embeddings.shape[1])
  difs = []
  
  #LIME-ATT
  lime_coefs = np.zeros(5)
  def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))
  
  sample = rs.normal( 0, 1, 500 * embeddings.shape[1]).reshape(500, embeddings.shape[1])*scale_emb + mean_emb
  sa   = annotate(sample)
  sam  = sa.mean(axis = 0)
  sasd = np.std(sa, axis = 0)
  anos_l = (sa - sam.reshape((1,-1)))/ sasd.reshape((1,-1))
  balance_l.append(Predict(sample).mean())
  weights = np.array([np.exp(-dist(x ,examples)**2/\
                     (np.sqrt(embeddings.shape[1])*.75)**2) for x in sample]) 
  lrl = Ridge(alpha = 0, fit_intercept = True, random_state = rs) 
  lrl.fit(anos_l, Predict_proba(sample), sample_weight = weights)

  coefssl   = np.vstack((coefssl,lrl.coef_))
  faith =   []
  balance = []
  dss     = []
  for k in fs:
    vertices = np.zeros(embeddings.shape[1])
    
    for i in range(5):
      
      vertices = np.vstack((vertices,
          create_positive_ver((z),i, k*d)))
      vertices = np.vstack((vertices,
          create_negative_ver((z),i, k*d)))
    print("Sampling radius = "  + str(k*d ))  

    vertices = (vertices[1:,:])
    print("Vertex shape: " + str(vertices.shape))
    print("Distance from boundary = " + str(d))
    def Simu(vers , N):
      samp = np.zeros((N, vers.shape[1]))
      m = vers.shape[0]
      for  i in range(N):
          u = np.append(np.insert(np.sort(rs.uniform(0,1,m)), 0, 0),1)
          w = [u[i] - u[j] for i,j in zip(range(1,u.shape[0]-1), range(u.shape[0]))]
          samp[i,:] = np.average(vers, weights = w, axis = 0)
      return samp 
    
    sim = Simu(vertices, 500)
    #Local standardization
    an = annotate(sim)
    pred_sim = Predict(sim)
    an_m = an.mean(axis = 0)
    an_std = np.std(an, axis = 0)
    anos = (an - an_m.reshape((1,-1)))/an_std.reshape((1,-1))

    
    lr = LogisticRegression(random_state = rs)
    lr.fit(anos, pred_sim)
    coef = n(lr.coef_)[0]
    
    
    #Tune the sampling radius
    if Predict(examples) == 1:
      
        a = examples - walk_db(coef, d + 0.1)
        print(Predict_proba(a))
        
        b = np.repeat((examples), a.shape[0], axis=0)
        mean_dist = 100000
        
        while mean_dist > 0.000001:

            mid = a/2 + b/2
            mid_pred = Predict((mid))
            same_class_as_example = mid_pred == ex_pred

            a[~same_class_as_example] = mid[~same_class_as_example]
            b[same_class_as_example] = mid[same_class_as_example]

            mean_dist = np.max(_dists(a, b))

        
        print("Prob :" + str(Predict_proba(a)))
        dista = (_dists(a, examples)[0])
          
    else:
        a = examples + walk_db(coef,d + 0.1)
        b = np.repeat((examples), a.shape[0], axis=0)

      
        mean_dist = 100000
        
        while mean_dist > 0.000001:

            mid = a/2 + b/2
            mid_pred = Predict((mid))
            same_class_as_example = mid_pred == ex_pred

            a[~same_class_as_example] = mid[~same_class_as_example]
            b[same_class_as_example] = mid[same_class_as_example]

            mean_dist = np.max(_dists(a, b))

        
        print("Rival prob:" + str(Predict_proba(a)))
        dista = (_dists(a, examples)[0])
        print("Distance from boundary: {} ".format(dista))

    difdb = np.copy(dista)
    if difdb < difdb1:
      coefs = n(lr.coef_)
      print(coefs)
      dss.append(difdb)
      samp  = np.copy(sim)
      
      faith.append((Predict(samp) == lr.predict(anos)).mean())
      
      print("Fidelity: {}".format((Predict(samp) == lr.predict(anos)).mean()))
      balance.append(Predict(samp).mean())
      difdb1 = np.copy(difdb)
      print("Picked, " + str(difdb))
  fig = plt.figure(figsize=(5, 5))  
  if Predict(examples) == 1:
    
    path_lime = [Predict_proba(examples - walk_db(n(lrl.coef_), (0.1+e))) for e in np.arange(0,3,.1)]
    path =[Predict_proba(examples - walk_db(n(coefs[0]), (0.1+e))) for e in np.arange(0,3,.1)] 
    plt.plot(np.arange(1,len(path_lime)+1), path, "blue")
    plt.plot(np.arange(1,len(path_lime)+1),  np.array(path_lime) , "green")
    
    #fig.savefig(wd +"/"+ "Figures" + "/" + str(idc) + "path_comparison_cel.png")
  else:
    path_lime = [Predict_proba(examples + walk_db(n(lrl.coef_), (0.1+e))) for e in np.arange(0,3,.1)]
    path =[Predict_proba(examples + walk_db(n(coefs[0]), (0.1+e))) for e in np.arange(0,3,.1)]  
    plt.plot(np.arange(1,len(path_lime)+1), path, "blue")
    plt.plot(np.arange(1,len(path_lime)+1),  np.array(path_lime) , "green")
    #fig.savefig(wd +"/"+ "Figures" + "/" + str(idc) + "path_comparison_cel.png")
  
  plt.clf()
  faiths.append(faith)
  ds.append(dss)
  balances.append(balance)
  faithg.append(lrl.score(anos_l, Predict(sample), sample_weight= weights))
  coefss4 = np.vstack((coefss4,coefs))
  ridge = LogisticRegression(penalty = "l1",C = .01, solver = "liblinear", random_state = rs)
  an = annotate(samp)
  pred_sim = Predict(samp)
  an_m = an.mean(axis = 0)
  an_std = np.std(an, axis = 0)
  anos = (an - an_m.reshape((1,-1)))/an_std.reshape((1,-1))
  ridge.fit(anos, pred_sim)
  coefss4l2 = np.vstack((coefss4l2 , n(ridge.coef_[0])))
  print("Optimal explanation vector: ")
  print(coefss4)
  
  mn = 16 

  nn = 15 
  gs1 = gridspec.GridSpec(3, 3)
  gs1.update(wspace=0.0, hspace=0.0)
  decoded_ex = decoder(examples).numpy()
  decoded_z = decoder(z.reshape((1,-1))).numpy()
  decoded_r = decoder(rivs[np.argsort(_dists(rivs,examples))][best].reshape((1,-1))).numpy()
  decoded_imgs = decoder(samp[mn:mn+nn,:]).numpy()


  fig = plt.figure(figsize=(5, 5))
  ax = plt.subplot(gs1[0])
  
  plt.imshow(c_batch[idc].reshape((128,128,3)))
  plt.title('$x_0$ {j}'.format(j = (Predict_proba(examples).round(2))), fontsize = 20)

  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  rect = patches.Rectangle((3.5,3),122.5,123,linewidth=2.5,edgecolor='r',facecolor='none')


  ax.add_patch(rect)
  ax = plt.subplot(gs1[1])
  plt.imshow(decoded_z.reshape((128,128,3)))
  plt.title("$x_b$ {i}".format(i = Predict_proba(z.reshape((1,-1))).round(2)), fontsize = 20)
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  rect = patches.Rectangle((3.5,3),122.5,123,linewidth=2.5,edgecolor='b',facecolor='none')


  ax.add_patch(rect)

  ax = plt.subplot(gs1[2])
  plt.imshow(decoded_r.reshape((128,128,3)))
  plt.title("$x_j$ {i}".format(i = (Predict_proba(rivs_ord[best].reshape((1,-1))).round(2))), fontsize = 20)
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
  rect = patches.Rectangle((3.5,3),122.5,123,linewidth=2.5,edgecolor='g',facecolor='none')


  ax.add_patch(rect)
  for i in range(3,9):
    
    ax = plt.subplot(gs1[i])
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    
  fig.savefig(wd +"/"+ "Figures" + "/" + str(idc) + "samples.png")
  plt.clf()

print("Saving results")
np.save("faiths_dba.npy", faiths)
np.save("faiths_lime.npy", faithg)
np.save("distances.npy", ds)
np.save("balances_dba.npy", balances)
np.save("balances_lime.npy", balance_l)
np.save("coefs_dba.npy", coefss4)
#np.save("coefs_lasso.npy", coefss4l2)
np.save("coefs_lime.npy", coefssl)
#np.save("sample.npy", samp)
#np.save("pred_samp.npy", Predict(samp))
print("Saved")

#CEM - MAF
print("Running CEM-MAF")
embeddings1 = code.predict(c_batch[to_explain])
import tensorflow as tf
def loss(dd, idx,ex_cll):
    def att_loss(dd, an):
      return 100*abs(an(dd)) + 100*tf.maximum(an(embeddings1[idx:idx+1]) - an(dd),tf.constant(0, dtype = "float32"))
    def att_loss_t(dd):
      a_loss = 0
      for A in annotators:
        a_loss += att_loss(dd , A)
        
      return a_loss  
    if ex_cll.numpy()[0][0] >= .5:    
          
      return tf.reduce_sum(tf.square(decoder(dd) - exampleim)) + \
      tf.reduce_sum(tf.square(embeddings1[idx:idx+1] - dd)) - \
      500*tf.minimum(1 - 2*cnn(decoder(dd)),5) + \
      att_loss_t(dd)
    else:
         
      return tf.reduce_sum(tf.square(decoder(dd) - exampleim)) + \
      tf.reduce_sum(tf.square(embeddings1[idx:idx+1] - dd)) - \
      500*tf.minimum(-1 + 2*cnn(decoder(dd)),5) + \
      att_loss_t(dd)

     
def grad(dd,idx,ex_cll):
    
    with tf.GradientTape() as tape:
        loss_value = loss(dd,idx,ex_cll)
    return tape.gradient(loss_value, [dd,])

coefs_cmaf = np.zeros((50,100))

for idxx in range(0,to_explain.shape[0]):
    
  adv_lat = tf.Variable(np.zeros(100).reshape((1,100)), dtype = "float32")
  example = tf.Variable(embeddings1[idxx:idxx+1].reshape((1,100)), dtype = "float32")
  exampleim = tf.Variable(c_batch[to_explain][idxx:idxx+1], dtype = "float32")
  cl = cnn(decoder(example))
  xxx = adv_lat.numpy()
  optimizer = tf.keras.optimizers.SGD(learning_rate=tf.keras.optimizers.schedules.PolynomialDecay(1e-2, 1000, 0, power= 0.5))

  
  for i in range(1000):
      grads = grad(adv_lat,idxx, cl)
      optimizer.apply_gradients(zip(grads, [adv_lat,]))
      if i % 20 == 0:
          print("Loss at step {:03d}: {:}".format(i, loss(adv_lat,idxx,cl)))
          
          xxx = adv_lat.numpy()
          
  coefs_cmaf[idxx,:] = xxx  

print("Saving CMAF results")  
np.save("coefs_cmaf.npy", coefs_cmaf)
print("Saved") 



print("Running latent space evaluation") 
i = 0

path_l = np.zeros((30,120))
path_db = np.zeros((30,120))
path_cmaf = np.zeros((30,120))

for idc in range(0,to_explain.shape[0]): 
  print(to_explain[idc])
  
  contrasts = annotate(embeddings1[idc:idc+1]) - annotate(coefs_cmaf[idc:idc+1])
  
  
  if Predict(embeddings1[idc:idc+1]) == 1:
    path_l[idc,:] = np.array([cnn.predict(decoder.predict(embeddings1[idc:idc+1] - walk_db(n(coefssl[1:][idc,:]) , (e))))[0][0] for e in np.arange(0,6,0.05)])
    path_db[idc,:] = np.array([cnn.predict(decoder.predict(embeddings1[idc:idc+1] - walk_db(n(coefss4[1:][idc,:]) , (e))))[0][0] for e in np.arange(0,6,0.05)])
    path_cmaf[idc,:] = np.array([cnn.predict(decoder.predict(embeddings1[idc:idc+1] - (e)* n(embeddings1[idc:idc+1] - coefs_cmaf[idc:idc+1])))[0][0] for e in np.arange(0,6,0.05)])
    #print(np.sum(path_cmaf[idc,:] - path_db[idc,:]))
    #print(np.sum((walk_db(n(coefssl[1:][idc,:]),0.05)**2)) - np.sum((walk_db(n(coefss4[1:][idc,:]),.05)**2)))
  else:
    path_l[idc,:] = np.array([cnn.predict(decoder.predict(embeddings1[idc:idc+1] + walk_db(n(coefssl[1:][idc,:]) , (e))))[0][0] for e in np.arange(0,6,0.05)])
    path_db[idc,:] = np.array([cnn.predict(decoder.predict(embeddings1[idc:idc+1] + walk_db(n(coefss4[1:][idc,:]) , (e))))[0][0] for e in np.arange(0,6,0.05)])
    path_cmaf[idc,:] = np.array([cnn.predict(decoder.predict(embeddings1[idc:idc+1] - (e)* n(embeddings1[idc:idc+1] - coefs_cmaf[idc:idc+1])))[0][0] for e in np.arange(0,6,0.05)])
    #print(np.sum(-path_cmaf[idc,:] + path_db[idc,:]))
    #print(np.sum((walk_db(n(coefssl[1:][idc,:]),.05)**2)) - np.sum((walk_db(n(coefss4[1:][idc,:]),.05)**2)))



    
  fig = plt.figure(figsize=(5, 5))
  xxx = np.arange(0,120)*0.05
  
  plt.plot(xxx, path_db[idc,:], c = "blue", lw = 3.2, label = "DBA-ATT")
  plt.plot(xxx, path_l[idc,:], c = "green" , lw = 3.2, label = "LIME-ATT")
  plt.plot(xxx,path_cmaf[idc,:], c = "red", lw = 3.2, label = "CEM-MAF")
  
  
  #plt.plot(path_gs[i,:], c = "magenta", lw = 3.2, label = "GS")
  
  plt.hlines(.5, -.1, 4, linestyles="dashed", label = "Decision \nboundary", colors = "red")
  
  plt.title("Latent Space Evaluation", size = 20)
  plt.yticks(size = 20)
  plt.xticks(size = 20)
  plt.legend(loc = 'upper center', bbox_to_anchor=(.9, 1.01), shadow=True, ncol=1 , fontsize = 17)
  #plt.xlabel(r"Distance from $\boldsymbol{x}_0$", size = 20,labelpad = 10)
  #plt.ylabel("CNN Response", size = 20, labelpad = 10)   
  #fig.savefig("latent_path_comparison_161_final.pdf", bbox_inches = "tight")  
  plt.xlim(0,4)
  #plt.legend(loc='upper center', bbox_to_anchor=(0.8, 0.9), shadow = True, ncol=1)
  plt.title("Latent Space Evaluation", size = 18)
  plt.xlabel(r"Distance from $x_0$", size = 20, labelpad =10)
  plt.ylabel("CNN Response", size = 20, labelpad = 10) 
  fig.savefig(wd +"/"+ "Figures" + "/" + str(to_explain[idc]) + "path_comparison_cel.png" , bbox_inches ="tight")  
  fig.clf() 
  plt.close()
  #gc.collect()
  fig = plt.figure(figsize=(5, 5))
  
  
  plt.imshow(c_batch[to_explain[idc]])
  plt.axis('off')
  
  fig.savefig(wd +"/"+ "Figures" + "/" + str(to_explain[idc]) + "true_image.png", bbox_inches = 'tight')  
  fig.clf() 
  plt.close()
 
 
def find_db_index(x):
  
  if x[0] < .5:
    for i in range(x.shape[0]):
      if i == x.shape[0]-1:
        return(i)
      if x[i] >= .5:
        return(i)
  else:
    for i in range(x.shape[0]):
      if i == x.shape[0]-1: 
        return(i)
      if x[i] <= 0.5:
        return(i)
        
cmaf_a = np.array([find_db_index(path_cmaf[j])*.05 for j in range(to_explain.shape[0])])
lime_a = np.array([find_db_index(path_l[j])*.05 for j in range(to_explain.shape[0])])
db_a = np.array([find_db_index(path_db[j])*.05 for j in range(to_explain.shape[0])])

avg_a_l  = lime_a.mean()
avg_a_db = db_a.mean()
avg_a_cmaf = cmaf_a.mean()


print("======= RESULTS ======================================")
print("Avg distance from decision boundary in Z (the smaller the better)")
print("------------------------------------")
print("DBA-ATT: {}".format(avg_a_db))
print("LIME-ATT: {}".format(avg_a_l))
print("CMAF: {}".format(avg_a_cmaf)) 

print("------------------------------------")
print("Fidelity for DBA and LIME")
faithsdb = np.array([x[-1] for x in faiths]).mean()
faithsl  = np.array(faithg).mean()
print("DBA fidelity:" + str(faithsdb))
print("LIME fidelity:" + str(faithsl))


print("Experiment completed succesfully. See Figures folder for latent space evaluation.")
      
