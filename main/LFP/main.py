from tkinter import FALSE
import pandas as pd
from SFS import Sequential_Forward_Selection_corr_test
from sklearn.linear_model import LinearRegression
from solve_loss import solve_loss_l5
from sklearn.model_selection import LeaveOneOut,cross_val_score
import numpy as np
import sklearn
import random
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, median_absolute_error


import warnings
import time
warnings.filterwarnings("ignore")


def modeling(csv_name,index,L,U,random_index_smote,random_index):
    """
    csv_name: name of the file of dataset
    L:number of labeled data
    U:number of unlabeled data
    random_index: a random sample of all the index of data
    repeat_num: random experiment index

    """

    data_samples = pd.read_csv(csv_name+'.csv',index_col=FALSE)
    data_shape = data_samples.shape
    data_columnslable_x = data_samples.columns[0:-2]
    data_columnslable_y = data_samples.columns[-2:-1]
    print(data_columnslable_y)

    print('labeled_num',L)
    data_labled = data_samples.loc[random_index[:L]]  
    data_unlabled = data_samples.loc[random_index[-U:]]
    

    print('unlabled_num',U)


    data_labled_x = data_labled[data_columnslable_x]
    data_labled_y = data_labled[data_columnslable_y]
    data_unlabled_x = data_unlabled[data_columnslable_x]
    result = np.loadtxt("./syn_data_KS/result"+str(index)+".csv",delimiter=",")
    data_sim = pd.DataFrame(result,columns=data_samples.columns)
    data_sim1 = data_sim[data_columnslable_x]
    data_unlabled_x_add = data_sim1.loc[random_index_smote]
 


    data_unlabled_y = data_unlabled[data_columnslable_y]
    ####Nomalize####
    mean_labled_x = data_labled_x.mean()
    std_labled_x = data_labled_x.std()

    mean_labled_y = data_labled_y.mean()
    std_labled_y = data_labled_y.std()


    X_train_labled = (data_labled_x-mean_labled_x)/std_labled_x

    X_train_unlabled = (data_unlabled_x-mean_labled_x)/std_labled_x
    X_train_unlabled_add = (data_unlabled_x_add-mean_labled_x)/std_labled_x
    y_train = (data_labled_y-mean_labled_y)/std_labled_y
    y_test = (data_unlabled_y-mean_labled_y)/std_labled_y
    #################

    ####build Partial model####
    Partial_feature_var2 = Sequential_Forward_Selection_corr_test(data_columnslable_x,data_labled_x,data_labled_y,random_index,L,L-2)
    Partial_feature = Partial_feature_var2[0]
    print('var features for'+str(L)+' '+'repeated_num '+str(index)+'is', Partial_feature)
 

    ZL = data_labled_x[Partial_feature]
    ZU = data_unlabled_x_add[Partial_feature]
    var2 = Partial_feature_var2[1]
    print('var2 is ',var2)

    ########################
    cv = LeaveOneOut()

    lasso_model = sklearn.linear_model.LassoCV(fit_intercept=False,cv=cv,alphas=[0.01,0.1,1,10])
    Reg_lasso = lasso_model.fit(X_train_labled.to_numpy(), y_train.to_numpy().ravel())

    ####find the var1####
    V = [0.5,1,2,5,10]
    print('current V',V)
    var1_candidate_set = [i*var2 for i in V] ##
    l5_candidate = [1,10,100]
    LOO_list = []

    for i in range(len(V)):
        for m in range(len(l5_candidate)):
            tmp_var1 = var1_candidate_set[i]
            l1 = 1/2/tmp_var1
            l2 = 1/2/var2
            l3 = 0
            l4 = 1/2/(tmp_var1+var2)
            #l5 = 0
            l5 = l1*l5_candidate[m]
            error_list = []
            for j in range(L):
                predict_x = data_labled_x.iloc[j]
                tmp_X = data_labled_x.drop(random_index[j])
                tmp_ZL = ZL.drop(random_index[j])
                predict_y = data_labled_y.iloc[j]
                tmp_y = data_labled_y.drop(random_index[j])

                ##Normalize##
                mean_labled_x = tmp_X.mean()
                std_labled_x = tmp_X.std()
                mean_labled_y = tmp_y.mean()
                std_labled_y = tmp_y.std()
                mean_labled_z = tmp_ZL.mean()
                std_labled_z = tmp_ZL.std()

                X_train_tmp = (tmp_X - mean_labled_x)/std_labled_x
                y_train_tmp = (tmp_y-mean_labled_y)/std_labled_y
                X_train_unlabled_tmp = (data_unlabled_x_add-mean_labled_x)/std_labled_x

                tmp_ZL = (tmp_ZL-mean_labled_z)/std_labled_z
                tmp_ZU = (ZU-mean_labled_z)/std_labled_z

                predict_x = (predict_x-mean_labled_x)/std_labled_x

                alpha,beta = solve_loss_l5(y_train_tmp.to_numpy(),X_train_tmp.to_numpy(),tmp_ZL.to_numpy(),
                        X_train_unlabled_tmp.to_numpy(),tmp_ZU.to_numpy(),l1,l2,l3,l4,l5)

                real_predict_y = predict_y.to_numpy()
                alpha_y = np.matmul(alpha.T,predict_x.to_numpy())
                real_alpha_y = (alpha_y*std_labled_y+mean_labled_y).to_numpy()


                tmp_error = (real_predict_y[0] - real_alpha_y[0])
                tmp_error_square = tmp_error * tmp_error
                error_list.append(tmp_error_square)
            LOO_list.append(np.mean(error_list))

    var1_index = np.argmin(np.array(LOO_list))
    print(var1_index)
    l1_index = var1_index // len(l5_candidate)
    print(l1_index)
    l5_index = var1_index % len(l5_candidate)
    print(l5_index) 
    #######################
    print('var1_index',var1_index)

    l1 = 1/2/var1_candidate_set[l1_index]
    l2 = 1/2/var2
    l3 = 0
    l4 = 1/2/(var1_candidate_set[l1_index]+var2)
    l5 = l1*l5_candidate[l5_index]

    ZL_all = X_train_labled[Partial_feature]
    ZU_all = X_train_unlabled[Partial_feature]
    ZU_all_add = X_train_unlabled_add[Partial_feature]

    alpha,beta = solve_loss_l5(y_train.to_numpy(),X_train_labled.to_numpy(),ZL_all.to_numpy(),
                        X_train_unlabled_add.to_numpy(),ZU_all_add.to_numpy(),l1,l2,l3,l4,l5)

    """
    Loss = (l1*math.pow(np.linalg.norm(y_train.to_numpy()-np.matmul(X_train_labled.to_numpy(),alpha)),2)+
            l2*math.pow(np.linalg.norm(y_train.to_numpy()-np.matmul(ZL.to_numpy(),beta)),2)+
            l3*math.pow(np.linalg.norm(np.matmul(X_train_labled.to_numpy(),alpha)-np.matmul(ZL.to_numpy(),beta)),2)+
            l4*math.pow(np.linalg.norm(np.matmul(X_train_unlabled.to_numpy(),alpha)-np.matmul(ZU.to_numpy(),beta)),2))
    """
    #check_loss(y_train,X_train_labled,ZL,
    #                    X_train_unlabled,ZU,l1,l2,l3,l4,alpha,beta,Loss)
    #print('Loss',math.pow(np.linalg.norm(np.matmul(X_train_unlabled.to_numpy(),alpha)-np.matmul(ZU.to_numpy(),beta)),2))
    #####################################


    ####start to test######

    test0_err_list = []
    err0_percent_list = []
    x_list = []
    y_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = np.matmul(beta.T,ZU_all.iloc[i].to_numpy())*std_labled_y+mean_labled_y
        real_y = tmp_y
        pre_y = tmp_pre
        x_list.append(real_y)
        y_list.append(pre_y)
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test0_err_list.append(tmp_error_square)
        err0_percent_list.append(tmp_percent_err)


    OURS_RMSE_log = np.sqrt(np.mean(test0_err_list))

    OURS_ERR = np.mean(err0_percent_list)
    OURS_MAE = mean_absolute_error(x_list,y_list)
    OURS_MAPE = mean_absolute_percentage_error(x_list,y_list)
    OURS_RMSE = root_mean_squared_error(x_list,y_list)

    test2_err_list = []
    err2_percent_list = []
    x_list = []
    y_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = Reg_lasso.predict([X_train_unlabled.iloc[i].to_numpy()])[0]*std_labled_y+mean_labled_y
        real_y = tmp_y
        pre_y = tmp_pre
        x_list.append(real_y)
        y_list.append(pre_y)
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test2_err_list.append(tmp_error_square)
        err2_percent_list.append(tmp_percent_err)
    print('LS_lasso_RMSE',np.sqrt(np.mean(test2_err_list)))
    print('LS_lasso_ERR',np.mean(err2_percent_list))
    LS_lasso_RMSE_log = np.sqrt(np.mean(test2_err_list))

    LS_lasso_ERR = np.mean(err2_percent_list)
    LS_lasso_MAE = mean_absolute_error(x_list,y_list)
    LS_lasso_MAPE = mean_absolute_percentage_error(x_list,y_list)
    LS_lasso_RMSE = root_mean_squared_error(x_list,y_list)

    l1_ratio_list = [0.1,0.3,0.5,0.7,0.9]
    elasticNet_model = sklearn.linear_model.ElasticNetCV(fit_intercept=False,cv=cv,alphas=[0.01,0.1,1,10],l1_ratio = l1_ratio_list)
    Reg_elasticnet = elasticNet_model.fit(X_train_labled.to_numpy(), y_train.to_numpy().ravel())
    print('alpha and l1_ratio of elasticNet is',Reg_elasticnet.alpha_,Reg_elasticnet.l1_ratio_)

    test3_err_list = []
    err3_percent_list = []
    x_list = []
    y_list = []
    for i in range(y_test.shape[0]):
        tmp_y = y_test.iloc[i].to_numpy()*std_labled_y+mean_labled_y
        tmp_pre = Reg_elasticnet.predict([X_train_unlabled.iloc[i].to_numpy()])[0]*std_labled_y+mean_labled_y
        real_y = tmp_y
        pre_y = tmp_pre
        x_list.append(real_y)
        y_list.append(pre_y)
        tmp_error = (tmp_y.to_numpy()[0] - tmp_pre.to_numpy()[0])
        tmp_percent_err =abs(tmp_error/tmp_y.to_numpy()[0])*100
        tmp_error_square = tmp_error * tmp_error
        test3_err_list.append(tmp_error_square)
        err3_percent_list.append(tmp_percent_err)
    print('LS_elasnet_RMSE',np.sqrt(np.mean(test3_err_list)))
    print('LS_elasnet_ERR',np.mean(err3_percent_list))
    LS_elasnet_RMSE_log = np.sqrt(np.mean(test3_err_list))

    LS_elasnet_ERR = np.mean(err3_percent_list)
    LS_elasnet_MAE = mean_absolute_error(x_list,y_list)
    LS_elasnet_MAPE = mean_absolute_percentage_error(x_list,y_list)
    LS_elasnet_RMSE = root_mean_squared_error(x_list,y_list)

    return OURS_MAPE,LS_lasso_MAPE,LS_elasnet_MAPE

if __name__=="__main__":
    repeated_num = 200
    all_average = {}
    all_median = {}
    csv_name = 'data_labeled'
    labeled_num = 10
    unlabeled_num = 30
    data_num = 40

    random_index = np.loadtxt('index.csv',  delimiter=',')
    random_index_smote = [random.sample(range(40),40) for i in range(repeated_num)]

    OURS_MAPE_all = {}
    ttt = range(5,30,5)
    for sweep_num in ttt:
        OURS_MAPE_all[sweep_num] = []

    for i in tqdm(range(repeated_num)):
    # for i in [9]:
        print('the ',i+1,'-th result')
        data_samples = pd.read_csv(csv_name+'.csv',index_col=FALSE)

        for sweep_num in ttt:
            start = time.time()

            OURS_MAPE,LS_lasso_MAPE,LS_elasnet_MAPE = modeling(csv_name,i,labeled_num,unlabeled_num,random_index_smote[i][:sweep_num],random_index[i])

            OURS_MAPE_all[sweep_num].append(OURS_MAPE)

            end = time.time()
            print('time for each trial: %s Seconds'%(end-start))


        pd.DataFrame(OURS_MAPE_all).to_csv('./OURS_MAPE_all.csv')