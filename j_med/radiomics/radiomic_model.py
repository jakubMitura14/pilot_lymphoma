import SimpleITK as sitk
import six
from radiomics import featureextractor, getTestCase
import multiprocessing
import optuna
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error
from ngboost.distns import Exponential, Normal
from ngboost import NGBClassifier
from ngboost.distns import k_categorical, Bernoulli
from mrmr import mrmr_classif
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import sklearn
import pickle
from pathlib import Path

# mrmr_selection,shap,ngboost


def display_probs(curr_class, inferred_probs, Y_test,to_be_sorted=True):

    probd_curr=inferred_probs[:,curr_class]
    class_curr=(Y_test==curr_class).to_numpy().astype(int)
    if(to_be_sorted):
        # Concatenate probd_curr and class_curr
        combined = np.column_stack((probd_curr, class_curr))

        # Sort by probd_curr
        combined_sorted = combined[combined[:, 0].argsort()[::-1]]

        # Divide back into probd_curr and class_curr
        probd_curr = combined_sorted[:, 0]
        class_curr = combined_sorted[:, 1]

    # Set the colors for the columns
    colors = ['red' if c == 0 else 'green' for c in class_curr]
    # Plot the column plot
    plt.bar(range(len(probd_curr)), probd_curr, color=colors)

    # Add a vertical line at value 0.5
    plt.axhline(y=0.5, color='blue', linestyle='--')



    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Probability')
    plt.title(f'class {curr_class}')

    # Show the plot
    plt.show()

def display_feature_importance(ngb,X_train):

    shap.initjs()

    ## SHAP plot for loc trees
    explainer = shap.TreeExplainer(ngb, model_output=0) # use model_output = 1 for scale trees
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=X_train.columns.to_numpy())



def get_tree_hyper_params(trial):
    criterion= "friedman_mse"#trial.suggest_categorical("criterion", ["squared_error", "friedman_mse", "absolute_error", "poisson"])
    splitter="random"#trial.suggest_categorical("splitter", ["best","random"])
    max_features=None#trial.suggest_categorical("max_features", ["sqrt","log2",None])
    # max_leaf_nodesint=trial.suggest_categorical("max_leaf_nodesint", [])
    max_depth=5#trial.suggest_int("max_depth", 1,10)
    min_samples_leaf=3#trial.suggest_int("min_samples_leaf", 1,3)
    min_impurity_decrease= 0.2307277162959608#trial.suggest_float("min_impurity_decrease", 0.0,0.3)

    return sklearn.tree.DecisionTreeRegressor(criterion=criterion,splitter=splitter,max_depth=max_depth,max_features=max_features,min_samples_leaf=min_samples_leaf
                                              ,min_impurity_decrease=min_impurity_decrease)





def clasify( main_df_val,main_df_train,y_cols,chosen_y_col,num_classes,K,to_display,Base,n_estimators,learning_rate,minibatch_frac):

    main_df_val[chosen_y_col]=main_df_val[chosen_y_col].to_numpy().astype(int)
    main_df_train[chosen_y_col]=main_df_train[chosen_y_col].to_numpy().astype(int)
    
    # main_df_val = main_df_val[main_df_val[chosen_y_col] > -1]
    # main_df_train = main_df_train[main_df_train[chosen_y_col] > -1]

    # print(f"mmmm val {len(main_df_val)} train {len(main_df_train)}")

    X_train = main_df_train.drop(columns=y_cols )
    X_test = main_df_val.drop(columns=y_cols)

    X_train = X_train.iloc[:, 1:]
    X_test = X_train.iloc[:, 1:]

    Y_train = main_df_train[chosen_y_col]
    Y_test = main_df_val[chosen_y_col]

    print(f"X_train {X_train} Y_train {Y_train}")

    # select top K features using mRMR
    selected_features = mrmr_classif(X=X_train, y=Y_train, K=7,n_jobs=1)
    # selected_features = mrmr_classif(X=None, y=Y_train, K=K)

    print(f"selected_features {selected_features}")
    # selected_features =['original_glcm_JointEntropy_adc', 'wavelet-HLH_firstorder_RobustMeanAbsoluteDeviation_adc', 'wavelet-LLL_firstorder_Kurtosis_adc', 'original_shape_Sphericity_adc', 'wavelet-LHL_firstorder_RootMeanSquared_hbv', 'original_glcm_SumEntropy_adc', 'log-sigma-3-0-mm-3D_glszm_SmallAreaEmphasis_adc']
    
    X_train=main_df_train[selected_features]
    X_test=main_df_val[selected_features]



    # # print(f"yyyyyyyyy {Y_train.to_numpy().astype(int)}")
    ngb_cat = NGBClassifier(Dist=k_categorical(num_classes), verbose=True
                            ,Base=Base
                            ,n_estimators=n_estimators
                            ,learning_rate=learning_rate
                            ,minibatch_frac=minibatch_frac) 
    # try:
    _ = ngb_cat.fit(X_train, Y_train.to_numpy().astype(int))
    # except:
    #     print(f"error")
    #     return 0.0
    
    file_path = Path('/workspaces/pilot_lymphoma/data/ngbtest.p')

    with file_path.open("wb") as f:
        pickle.dump(ngb_cat, f)

    # with file_path.open("rb") as f:
    #     ngb_cat = pickle.load(f)

    if(to_display):
        #display feature importance
        display_feature_importance(ngb_cat,X_train)


    inferred=ngb_cat.predict(X_test)
    # print(f"iii {inferred}")
    # print(f"iii2 {Y_test.to_numpy()}")

    acc=accuracy_score(Y_test.to_numpy(), inferred)
    inferred_probs = ngb_cat.predict_proba(X_test)

    if(to_display):
        for curr_class in range(num_classes):
            display_probs(curr_class, inferred_probs, Y_test)

    # print(f"probs {inferred_probs}")
    print(f"""Accuracy: {acc}""")
    if(num_classes==2):
        a=(inferred_probs[:,1]>0.7).astype(bool)
        b=Y_test.to_numpy()
        high_confidence=np.sum(np.logical_and(a,b).flatten())/np.sum(b.flatten())
        print(f"high_confidence {high_confidence}")








    return acc
#K is number of features we want to select
# K=20

# def classify_full(trial):
def classify_full():
    
    # K=20
    K=7
    # X, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
    res_path=""
    main_df=pd.read_csv("/workspaces/pilot_lymphoma/data/extracted_features_pet2.csv")

    # Get first 20 percent of rows
    main_df_val = main_df.head(int(len(main_df) * 0.2))
    main_df_train = main_df.tail(int(len(main_df) * 0.8))

    y_cols=["pat_id","lesion_num","study_0_or_1","Deauville","lab_path","mod_name"]#,"vol_in_mm3"
    # clinical_cols=["dre","psa","age"]
    # clinical_cols=["psa","age","dre"]
    # chosen_y_col="is_cancer"
    # chosen_y_col="isup"
    # chosen_y_col="isup_simple"
    chosen_y_col="Deauville"
    # num_classes=2
    num_classes=6

    n_estimators=866#trial.suggest_int("n_estimators", 100,2000)   
    learning_rate=0.02639867572400997#trial.suggest_float("learning_rate", 0.00001,0.1)   
    minibatch_frac = 0.7561751607203051#trial.suggest_float("minibatch_frac", 0.7,1.0) 

   
    # clasify( main_df_val,main_df_train,y_cols,clinical_cols,chosen_y_col,num_classes,K)
    # clasify( main_df_val,main_df_train,y_cols,clinical_cols,chosen_y_col,num_classes,K)
    Base=get_tree_hyper_params([])    
    res=clasify( main_df_val,main_df_train,y_cols,chosen_y_col,num_classes,K,False,Base,n_estimators,learning_rate,minibatch_frac)
    return res


    # in case of clasyfing isup we need to take a maximum of the isup values for each lesion

classify_full()

# database_name="nat"
# experiment_name="nat_199"
# # storage = optuna.storages.RDBStorage(
# #     url=f"mysql://root@34.90.134.17/{database_name}",
# #     # engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
# # )

# study = optuna.create_study(
#         study_name=experiment_name
#         # ,sampler=optuna.samplers.CmaEsSampler()    
#         ,sampler=optuna.samplers.NSGAIISampler()    
#         # ,pruner=optuna.pruners.HyperbandPruner()
#         # ,storage=f"mysql://root:jm@34.90.134.17:3306/{experiment_name}"
#         # ,storage=f"mysql://root@34.90.134.17/{database_name}"
#         # ,load_if_exists=True
#         ,direction="maximize"
#         )

# study.optimize(classify_full, n_trials=90000,gc_after_trial=True)