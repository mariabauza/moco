import cv2, os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


font_size = 30

#plt.rc('font', size=font_size)          # controls default text sizes                                                          \
                                                                                                                                
plt.rc('axes', titlesize=font_size)     # fontsize of the axes title                                                           \
                                                                                                                                
plt.rc('axes', labelsize=font_size)    # fontsize of the x and y labels                                                        \
                                                                                                                                
plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels                                                          \
                                                                                                                                
plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels                                                          \
                                                                                                                                
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize                                                                    \
                                                                                                                                
plt.rc('figure', titlesize=35)  # fontsize of the figure title    



object_names = ['pin_view2'] #, 'grease_view1', 'head_view1', 'curved_view1']  #'pin_view2_v2'
num_epoch = 200
def plot_violin(object_name, grid_name, model_dir):
  errors = np.load('{}.npy'.format(object_name), allow_pickle=True)
  #9closest_ICP_errors, 10match_errors_v2,11best_dist_errors_v2,closest_errors_v2, NN_ICP_errors, best_NN_ICP_dist_errors,pix_errors, best_pix_dist_errors, best_ICP_5_dist_errors, best_ICP_1_dist_errors ])

  random_mean = 1 #np.mean(errors[0])
  errors /= random_mean

  print("Error shape",errors.shape) # 18 features by number of data points 
  for j in np.arange(9,num_epoch, 10):
      print(j)
      best1 = errors[1] 
      
      ICP1 = errors[4] 
      best10 = errors[2] 
      ICP10 = errors[17]
      path_data = '/home/mcube/moco/data/{}_{}/models/{}/'.format(object_name, grid_name,model_dir)
      
      ICP1 = np.load(path_data +'errors_checkpoint={}_is_test=True_is_real=True_is_detectron2=False.npy'.format(str(j).zfill(4))).flatten()*1000
      ICP10 = np.load(path_data +'errors_checkpoint={}_is_test=True_is_real=False_is_detectron2=False.npy'.format(str(j).zfill(4))).flatten()*1000
      
      ICP1 /= random_mean
      ICP10 /= random_mean
      print('Original median:', np.median(best1), 'Current median:', np.median(ICP1), 'With true, median:', np.median(ICP10))
      
      
      #column_names = ["pose_error", "xlab", "icp"]
      #best1_df = pd.DataFrame(columns = column_names)
      #ICP1_df = pd.DataFrame(columns = column_names)
      #best10_df = pd.DataFrame(columns = column_names)
      #ICP10_df = pd.DataFrame(columns = column_names)

      best1_df = pd.DataFrame({'pose_error':best1}) 
      ICP1_df = pd.DataFrame({'pose_error':ICP1})
      best10_df = pd.DataFrame({'pose_error':best10}) 
      ICP10_df = pd.DataFrame({'pose_error':ICP10})

      #making sure aech dataset is labeled so that htey take different positions on the x axis 
      num_data = errors.shape[1]

      # best = 1
      # best 10 = 3
      best1_df['xlab'] = pd.DataFrame(np.ones(num_data)) 
      ICP1_df['xlab'] = pd.DataFrame(np.ones(num_data))
      best10_df['xlab'] = pd.DataFrame(3*np.ones(num_data)) 
      ICP10_df['xlab'] = pd.DataFrame(3*np.ones(num_data))

      # giveing the label for the hue so that they take the same spot but are plotted seperately
      # without ICP = 1
      # with ICP = 3
      best1_df['icp'] = pd.DataFrame(np.ones(num_data)) 
      ICP1_df['icp'] = pd.DataFrame(3*np.ones(num_data))
      best10_df['icp'] = pd.DataFrame(np.ones(num_data)) 
      ICP10_df['icp'] = pd.DataFrame(3*np.ones(num_data))

      #combine everything into noe dataset
      full_df = best1_df.append(ICP1_df)
      #print(full_df.shape)
      full_df = full_df.append(best10_df)
      #print(full_df.shape)
      full_df = full_df.append(ICP10_df)
      #print(full_df.shape)

      print(full_df.columns)

      #flatui = ["#3498db", "#2ecc71"]
      #flatui = ["#1A35A4", "#20EE20"]
      flatui = ["#4C6EE6", "#41B773"]
      #sns.palplot(sns.color_palette(flatui))
      
      fig, ax = plt.subplots(constrained_layout=True)
      fig.set_size_inches(10, 8)
      
      sns.violinplot(x="xlab", y='pose_error', hue="icp",
                     data=full_df, palette=flatui, split=True, inner="quartile", bw=0.075, linewidth=2, scale="area")

      bottom, top = plt.ylim()
      plt.yticks(np.arange(0,top, 0.25))

      ax2 = plt.twinx()
      axes = plt.gca()
      mn, mx = axes.get_ylim()
      ax.set_ylim(0, top, 0.25)
      ax2.set_ylim(0, top*random_mean)

      ax.set_ylabel('Normalized Pose Error', labelpad=2)
      ax2.set_ylabel('Pose Error (mm)')

      #plt.ylabel('Pose error (mm)', labelpad=10)
      #plt.xlabel('Number of contacts')
      plt.title('{}'.format(object_name))
      plt.ylim(bottom=0)
      plt.xlim([-0.4,1.5])
      #plt.show()


      stds = [];   means = [];   medians = []
      for it_err in range(errors.shape[0]):
        error = errors[it_err] #iterating through each feature
        #print(errors.shape)
        if it_err in [3, 5, 6,8,9,10,11,12,13,14,15,16]: continue  #Errors we do not need
        stds.append(np.std(error))
        medians.append(np.median(error))
        means.append(np.mean(error))
      
      means = np.array(means)
      medians = np.array(medians)
      stds = np.array(stds)

      #print("medain shape:", medians.shape)
      
      width = 0.35       # the width of the bars: can also be len(x) sequence

      ind_order = [1,3,2,5]
      ind = np.arange(len(ind_order)) 
      #N = len(ind_order)
      Nstart = -0.35
      Nstop = 1.45
      N = np.arange(Nstart,Nstop,0.1)
      #p1 = plt.bar(ind, medians[ind_order], width)
      #p2 = plt.bar(ind, means[ind_order]-medians[ind_order], width, bottom=medians[ind_order]) #, yerr=stds[ind_order])
      pr = plt.plot(np.arange(Nstart,Nstop,0.1), [means[0]*random_mean]*np.ones(len(N)), 'k', linewidth=5, solid_capstyle='round')
      pc = plt.plot(np.arange(Nstart,Nstop,0.1), [means[4]*random_mean]*np.ones(len(N)), '#C44E52', linewidth=5, solid_capstyle='round')
      #plt.ylabel('Pose error')
      #plt.title('Pose error for {}'.format(object_name))
      #plt.xticks(ind, ('Best1', 'ICP1', 'Best10', 'ICP10'))

      #plt.legend((p1[0], p2[0], pr[0], pc[0]), ('Mean', 'Median', 'Close', 'Random'))

      plt.savefig(path_data + 'pose_results_epoch={}.png'.format(j))
      #plt.show()


