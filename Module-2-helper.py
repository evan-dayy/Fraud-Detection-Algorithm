
import os
import pandas as pd
import numpy as np
import math
import sys
import time
import pickle
import json
import datetime
import random
import sklearn
from sklearn import *
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import xgboost
import imblearn
sns.set_style('whitegrid', {'axes.facecolor': '0.8'})
# --------------------------------------------------------------------------------- Feature Engineering
# ----------------------------------------------------
# Binary Output : Whether a day is during weekend or during weekday
def is_weekend(tx_datetime):
    weekday = tx_datetime.weekday()
    is_weekend = weekday>=5
    return int(is_weekend)

# ----------------------------------------------------
# Binary Output: Whether the transaction happens during night
def is_night(tx_datetime):
    tx_hour = tx_datetime.hour
    is_night = tx_hour<=8
    return int(is_night)

# ----------------------------------------------------
# define a function computing the average transaction amount in each window size (Customer Views)
def compute_avg_amt(C_T, window):
    for window_size in window:
        # Compute the SUM
        _SUM = C_T['Trans_AMOUNT'].rolling(str(window_size)+'d').sum()
        _WIND = C_T['Trans_AMOUNT'].rolling(str(window_size)+'d').count()
        # Compute the AVG
        _AVG = _SUM/_WIND
        # Saving
        C_T['WIND_Trans_'+str(window_size)+'DAY']=list(_WIND)
        C_T['AVG_AMOUNT_'+str(window_size)+'DAY']=list(_AVG)

def get_customer_spending_behaviour_features(C_T, window=[1,7,30]):
    # Order transactions chronologically
    C_T=C_T.sort_values('Trans_DATETIME')
    C_T.index=C_T.Trans_DATETIME
    compute_avg_amt(C_T, window)
    # Reindex according to transaction IDs
    C_T.index=C_T.TRANSACTION_ID
    # And return the dataframe with the new features
    return C_T

# ----------------------------------------------------
# define a function computing the average transaction amount in each window size (STORE Views)
def update_features(store_T, delay_period, window, feature, NB_FRAUD_DELAY, NB_Trans_DELAY):
    for window_size in window:
        NB_FRAUD=store_T['Trans_FRAUD'].rolling(str(delay_period+window_size)+'d').sum()
        NB_DELAY=store_T['Trans_FRAUD'].rolling(str(delay_period+window_size)+'d').count()
        NB_FRAUD_WINDOW=NB_FRAUD-NB_FRAUD_DELAY
        NB_Trans_WINDOW=NB_DELAY-NB_Trans_DELAY
        RISK_WINDOW=NB_FRAUD_WINDOW/NB_Trans_WINDOW
        store_T[feature+'_NB_Trans_'+str(window_size)+'DAY_WINDOW']=list(NB_Trans_WINDOW)
        store_T[feature+'_RISK_'+str(window_size)+'DAY_WINDOW']=list(RISK_WINDOW)
        
def get_count_risk_rolling_window(store_T, delay_period=7, window=[1,7,30], feature="STORE_ID"):
    store_T=store_T.sort_values('Trans_DATETIME')
    store_T.index=store_T.Trans_DATETIME
    NB_FRAUD_DELAY=store_T['Trans_FRAUD'].rolling(str(delay_period)+'d').sum()
    NB_Trans_DELAY=store_T['Trans_FRAUD'].rolling(str(delay_period)+'d').count()
    update_features(store_T, delay_period, window, feature, NB_FRAUD_DELAY, NB_Trans_DELAY)
    store_T.index=store_T.TRANSACTION_ID
    # Replace NA values with 0 (all undefined risk scores where NB_Trans_WINDOW is 0) 
    store_T.fillna(0,inplace=True)
    return store_T

# --------------------------------------------------------------------------------- Model Building
# ----------------------------------------------------
# reading the data from the previous sections
def read_from_files(directory, start, end):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f>=start+'.pkl' and f<=end+'.pkl']
    frames = []
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
    df_final = pd.concat(frames)
    df_final=df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True,inplace=True)
    df_final=df_final.replace([-1],0) # marking the missing values
    return df_final

# ----------------------------------------------------
def get_train_test_set(transactions_df, start_date_training,
                       delta_train=7,
                       delta_delay=7,
                       delta_test=7,
                       sampling_ratio=1, random_state=33):
    train_df = transactions_df[(transactions_df.Trans_DATETIME>=start_date_training) & (transactions_df.Trans_DATETIME<start_date_training+datetime.timedelta(days=delta_train))]
    test_df = []
    # Generating the valid test set
    # First,defrauded customer ID
    known_defrauded_customers = set(train_df[train_df.Trans_FRAUD==1].CUSTOMER_ID)
    # Find the relative start day
    start_tx_time_days_training = train_df.Trans_TIME_DAYS.min()
    # Looping to find the compromised card, and considering the delay period
    for day in range(delta_test):
        test_df_day = transactions_df[transactions_df.Trans_TIME_DAYS==start_tx_time_days_training+delta_train+delta_delay+day]
        test_df_day_delay_period = transactions_df[transactions_df.Trans_TIME_DAYS==start_tx_time_days_training+delta_train+day-1]
        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.Trans_FRAUD==1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        test_df.append(test_df_day)
    test_df = pd.concat(test_df)
    # This is the section because of resampling issues for imbalance classes
    # If subsample
    if sampling_ratio<1:
        train_df_frauds=train_df[train_df.Trans_FRAUD==1].sample(frac=sampling_ratio, random_state=random_state)
        train_df_genuine=train_df[train_df.Trans_FRAUD==0].sample(frac=sampling_ratio, random_state=random_state)
        train_df=pd.concat([train_df_frauds,train_df_genuine])
    # Sort data sets by ascending order of transaction ID
    train_df=train_df.sort_values('TRANSACTION_ID')
    test_df=test_df.sort_values('TRANSACTION_ID')
    return (train_df, test_df)

# ---------------------------------------------------- Cross Validation
def prequentialSplit(transactions_df,
                     start_date_training, 
                     n_folds=4, 
                     delta_train=7,
                     delta_delay=7,
                     delta_assessment=7):
    prequential_split_indices=[]
    # Start an iteration to go throught all the folds
    for fold in range(n_folds):
        start_date_training_fold = start_date_training-datetime.timedelta(days=fold*delta_assessment)
        # Updating the train and test data sets
        (train_df, test_df)=get_train_test_set(transactions_df, start_date_training=start_date_training_fold,
                                               delta_train=delta_train,delta_delay=delta_delay,delta_test=delta_assessment)
        indices_train=list(train_df.index)
        indices_test=list(test_df.index)
        prequential_split_indices.append((indices_train,indices_test))
    return prequential_split_indices

def prequential_grid_search(transactions_df, 
                            classifier, 
                            input_features, output_feature, 
                            parameters, scoring, 
                            start_date_training, 
                            n_folds=4,
                            expe_type='Test',
                            delta_train=7, 
                            delta_delay=7, 
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            n_jobs=-1):
    estimators = [('scaler', sklearn.preprocessing.StandardScaler()), ('clf', classifier)]
    pipe = sklearn.pipeline.Pipeline(estimators)
    prequential_split_indices=prequentialSplit(transactions_df,
                                               start_date_training=start_date_training, 
                                               n_folds=n_folds, 
                                               delta_train=delta_train, 
                                               delta_delay=delta_delay, 
                                               delta_assessment=delta_assessment)
    grid_search = sklearn.model_selection.GridSearchCV(pipe, parameters, scoring=scoring, cv=prequential_split_indices, refit=False, n_jobs=n_jobs)
    X, y = transactions_df[input_features], transactions_df[output_feature]
    grid_search.fit(X, y)
    performances_df=pd.DataFrame()
    for i in range(len(performance_metrics_list_grid)):
        performances_df[performance_metrics_list[i]+' '+expe_type]=grid_search.cv_results_['mean_test_'+performance_metrics_list_grid[i]]
        performances_df[performance_metrics_list[i]+' '+expe_type+' Std']=grid_search.cv_results_['std_test_'+performance_metrics_list_grid[i]]
    performances_df['Parameters']=grid_search.cv_results_['params']
    performances_df['Execution time']=grid_search.cv_results_['mean_fit_time']
    return performances_df

# ----------------------------------------------------
def model_selection_wrapper(transactions_df, 
                            classifier, 
                            input_features, output_feature,
                            parameters, 
                            scoring, 
                            start_date_training_for_valid,
                            start_date_training_for_test,
                            n_folds=4,
                            delta_train=7, 
                            delta_delay=7, 
                            delta_assessment=7,
                            performance_metrics_list_grid=['roc_auc'],
                            performance_metrics_list=['AUC ROC'],
                            n_jobs=-1):
    # Performance on the Valid Sets
    performances_df_validation=prequential_grid_search(transactions_df, classifier, 
                            input_features, output_feature,
                            parameters, scoring, 
                            start_date_training=start_date_training_for_valid,
                            n_folds=n_folds,
                            expe_type='Validation',
                            delta_train=delta_train, 
                            delta_delay=delta_delay, 
                            delta_assessment=delta_assessment,
                            performance_metrics_list_grid=performance_metrics_list_grid,
                            performance_metrics_list=performance_metrics_list,
                            n_jobs=n_jobs)
    # Performance on test set
    performances_df_test=prequential_grid_search(transactions_df, classifier, 
                            input_features, output_feature,
                            parameters, scoring, 
                            start_date_training=start_date_training_for_test,
                            n_folds=n_folds,
                            expe_type='Test',
                            delta_train=delta_train, 
                            delta_delay=delta_delay, 
                            delta_assessment=delta_assessment,
                            performance_metrics_list_grid=performance_metrics_list_grid,
                            performance_metrics_list=performance_metrics_list,
                            n_jobs=n_jobs)
    # Output
    performances_df_validation.drop(columns=['Parameters','Execution time'], inplace=True)
    performances_df=pd.concat([performances_df_test,performances_df_validation],axis=1)
    return performances_df

# ----------------------------------------------------
def get_summary_performances(performances_df, parameter_column_name="Parameters summary"):
    # Performance metrics
    metrics = ['AUC ROC','Average precision','Card Precision@100']
    performances_results=pd.DataFrame(columns=metrics)
    performances_df.reset_index(drop=True,inplace=True)
    best_estimated_parameters = []
    validation_performance = []
    test_performance = []
    for metric in metrics:
        index_best_validation_performance = performances_df.index[np.argmax(performances_df[metric+' Validation'].values)]
        best_estimated_parameters.append(performances_df[parameter_column_name].iloc[index_best_validation_performance])
        validation_performance.append(
                str(round(performances_df[metric+' Validation'].iloc[index_best_validation_performance],3))+
                '+/-'+
                str(round(performances_df[metric+' Validation'+' Std'].iloc[index_best_validation_performance],2))
        )
        test_performance.append(str(round(performances_df[metric+' Test'].iloc[index_best_validation_performance],3))+
                '+/-' + str(round(performances_df[metric+' Test'+' Std'].iloc[index_best_validation_performance],2)))
    performances_results.loc["Best estimated parameters"]=best_estimated_parameters
    performances_results.loc["Validation performance"]=validation_performance
    performances_results.loc["Test performance"]=test_performance
    optimal_test_performance = []
    optimal_parameters = []
    for metric in ['AUC ROC Test','Average precision Test','Card Precision@100 Test']:
        # provides the optimal performance
        index_optimal_test_performance = performances_df.index[np.argmax(performances_df[metric].values)]
        # Retrieve 
        optimal_parameters.append(performances_df[parameter_column_name].iloc[index_optimal_test_performance])
        # Add test performance to the test_performance list (mean+/-std)
        optimal_test_performance.append(str(round(performances_df[metric].iloc[index_optimal_test_performance],3))+
                '+/-'+ str(round(performances_df[metric+' Std'].iloc[index_optimal_test_performance],2)))
    # Add results to the performances_results DataFrame
    performances_results.loc["Optimal parameters"]=optimal_parameters
    performances_results.loc["Optimal test performance"]=optimal_test_performance
    return performances_results

# ----------------------------------------------------
def get_performance_plot(performances_df, 
                         ax, 
                         performance_metric, 
                         expe_type_list=['Test','Train'], 
                         expe_type_color_list=['#008000','#2F4D7E'],
                         parameter_name="Tree maximum depth",
                         summary_performances=None):
    for i in range(len(expe_type_list)):
        performance_metric_expe_type=performance_metric+' '+expe_type_list[i]
        # Plot
        ax.plot(performances_df['Parameters summary'], performances_df[performance_metric_expe_type], 
                color=expe_type_color_list[i], label = expe_type_list[i])
        # If performances_df contains confidence intervals, add them to the graph
        if performance_metric_expe_type+' Std' in performances_df.columns:
            conf_min = performances_df[performance_metric_expe_type]-2*performances_df[performance_metric_expe_type+' Std']
            conf_max = performances_df[performance_metric_expe_type]+2*performances_df[performance_metric_expe_type+' Std']
            ax.fill_between(performances_df['Parameters summary'], conf_min, conf_max, color=expe_type_color_list[i], alpha=.1)
    # If summary_performances table is present, adds vertical dashed bar for best estimated parameter 
    if summary_performances is not None:
        best_estimated_parameter=summary_performances[performance_metric][['Best estimated parameters']].values[0]
        best_estimated_performance=float(summary_performances[performance_metric][['Validation performance']].values[0].split("+/-")[0])
        ymin, ymax = ax.get_ylim()
        ax.vlines(best_estimated_parameter, ymin, best_estimated_performance,
                  linestyles="dashed")
    # Set title, and x and y axes labels
    ax.set_title(performance_metric+'\n', fontsize=14)
    ax.set(xlabel = parameter_name, ylabel=performance_metric)

# ----------------------------------------------------
def get_performances_plots(performances_df, 
                           performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                           expe_type_list=['Test','Train'], expe_type_color_list=['#008000','#2F4D7E'],
                           parameter_name="Tree maximum depth",
                           summary_performances=None):
    n_performance_metrics = len(performance_metrics_list)
    fig, ax = plt.subplots(1, n_performance_metrics, figsize=(5*n_performance_metrics,4))
    for i in range(n_performance_metrics):
        get_performance_plot(performances_df, ax[i], performance_metric=performance_metrics_list[i], 
                             expe_type_list=expe_type_list, 
                             expe_type_color_list=expe_type_color_list,
                             parameter_name=parameter_name,
                             summary_performances=summary_performances)
    ax[n_performance_metrics-1].legend(loc='upper left', labels=expe_type_list, bbox_to_anchor=(1.05, 1),title="Type set")
    plt.subplots_adjust(wspace=0.5, hspace=0.8)
    

# ----------------------------------------------------
def card_precision_top_k_day(df_day,top_k):
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(drop=False)
    # Get the top k most suspicious cards
    df_day_top_k=df_day.head(top_k)
    list_detected_compromised_cards=list(df_day_top_k[df_day_top_k.Trans_FRAUD==1].CUSTOMER_ID)
    # Compute precision top k
    card_precision_top_k = len(list_detected_compromised_cards) / top_k
    return list_detected_compromised_cards, card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):
    # Sort days by increasing order
    list_days=list(predictions_df['Trans_TIME_DAYS'].unique())
    list_days.sort()
    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []
    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []
    # For each day, compute precision top k
    for day in list_days:
        df_day = predictions_df[predictions_df['Trans_TIME_DAYS']==day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'Trans_FRAUD']]
        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards)==False]
        nb_compromised_cards_per_day.append(len(df_day[df_day.Trans_FRAUD==1].CUSTOMER_ID.unique()))
        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(df_day,top_k)
        card_precision_top_k_per_day_list.append(card_precision_top_k)
        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)
    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()
    # Returns precision top k per day as a list, and resulting mean
    return nb_compromised_cards_per_day,card_precision_top_k_per_day_list,mean_card_precision_top_k


def card_precision_top_k_custom(y_true, y_pred, top_k, transactions_df):
    # Let us create a predictions_df DataFrame, that contains all transactions matching the indices of the current fold
    # (indices of the y_true vector)
    predictions_df=transactions_df.iloc[y_true.index.values].copy()
    predictions_df['predictions']=y_pred
    # Compute the CP@k using the function implemented in Chapter 4, Section 4.2
    nb_compromised_cards_per_day,card_precision_top_k_per_day_list,mean_card_precision_top_k=card_precision_top_k(predictions_df, top_k)
    # Return the mean_card_precision_top_k
    return mean_card_precision_top_k

# ----------------------------------------------------
def get_execution_times_plot(performances_df,
                             title="",
                             parameter_name="Tree maximum depth"):
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    # Plot data on graph
    ax.plot(performances_df['Parameters summary'], performances_df["Execution time"], color="black")
    # Set title, and x and y axes labels
    ax.set_title(title, fontsize=14)
    ax.set(xlabel = parameter_name, ylabel="Execution time (seconds)")
    
    
def model_selection_performances(performances_df_dictionary,
                                 performance_metric='AUC ROC',
                                 model_classes=['Decision Tree', 
                                                'Logistic Regression', 
                                                'Random Forest', 
                                                'XGBoost'],
                                 default_parameters_dictionary={
                                                "Decision Tree": 50,
                                                "Logistic Regression": 1,
                                                "Random Forest": "100/50",
                                                "XGBoost": "100/0.3/6"
                                            }):
    
    mean_performances_dictionary={
        "Default parameters": [],
        "Best validation parameters": [],
        "Optimal parameters": []
    }
    
    std_performances_dictionary={
        "Default parameters": [],
        "Best validation parameters": [],
        "Optimal parameters": []
    }
    # For each model class
    for model_class in model_classes:
        performances_df=performances_df_dictionary[model_class]
        # Get the performances for the default paramaters
        default_performances=performances_df[performances_df['Parameters summary']==default_parameters_dictionary[model_class]]
        default_performances=default_performances.round(decimals=3)
        mean_performances_dictionary["Default parameters"].append(default_performances[performance_metric+" Test"].values[0])
        std_performances_dictionary["Default parameters"].append(default_performances[performance_metric+" Test Std"].values[0])
        # Get the performances for the best estimated parameters
        performances_summary=get_summary_performances(performances_df, parameter_column_name="Parameters summary")
        mean_std_performances=performances_summary.loc[["Test performance"]][performance_metric].values[0]
        mean_std_performances=mean_std_performances.split("+/-")
        mean_performances_dictionary["Best validation parameters"].append(float(mean_std_performances[0]))
        std_performances_dictionary["Best validation parameters"].append(float(mean_std_performances[1]))
        # Get the performances for the boptimal parameters
        mean_std_performances=performances_summary.loc[["Optimal test performance"]][performance_metric].values[0]
        mean_std_performances=mean_std_performances.split("+/-")
        mean_performances_dictionary["Optimal parameters"].append(float(mean_std_performances[0]))
        std_performances_dictionary["Optimal parameters"].append(float(mean_std_performances[1]))
    # Return the mean performances and their standard deviations    
    return (mean_performances_dictionary,std_performances_dictionary)


# ----------------------------------------------------
# Get the performance plot for a single performance metric
def get_model_selection_performance_plot(performances_df_dictionary, 
                                         ax, 
                                         performance_metric,
                                         ylim=[0,1],
                                         model_classes=['Decision Tree', 
                                                        'Logistic Regression', 
                                                        'Random Forest', 
                                                        'XGBoost']):
    (mean_performances_dictionary,std_performances_dictionary) = \
        model_selection_performances(performances_df_dictionary=performances_df_dictionary,
                                     performance_metric=performance_metric)
    
    # width of the bars
    barWidth = 0.3
    # The x position of bars
    r1 = np.arange(len(model_classes))
    r2 = r1+barWidth
    r3 = r1+2*barWidth
    # Create Default parameters bars (Orange)
    ax.bar(r1, mean_performances_dictionary['Default parameters'], 
           width = barWidth, color = '#CA8035', edgecolor = 'black', 
           yerr=std_performances_dictionary['Default parameters'], capsize=7, label='Default parameters')
 
    # Create Best validation parameters bars (Red)
    ax.bar(r2, mean_performances_dictionary['Best validation parameters'], 
           width = barWidth, color = '#008000', edgecolor = 'black', 
           yerr=std_performances_dictionary['Best validation parameters'], capsize=7, label='Best validation parameters')

    # Create Optimal parameters bars (Green)
    ax.bar(r3, mean_performances_dictionary['Optimal parameters'], 
           width = barWidth, color = '#2F4D7E', edgecolor = 'black', 
           yerr=std_performances_dictionary['Optimal parameters'], capsize=7, label='Optimal parameters')
    # Set title, and x and y axes labels
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xticks(r2+barWidth/2)
    ax.set_xticklabels(model_classes, rotation = 45, ha="right", fontsize=12)
    ax.set_title(performance_metric+'\n', fontsize=18)
    ax.set_xlabel("Model class", fontsize=16)
    ax.set_ylabel(performance_metric, fontsize=15)
    
    
# ----------------------------------------------------
def get_model_selection_performances_plots(performances_df_dictionary, 
                                           performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'],
                                           ylim_list=[[0.6,0.9],[0.2,0.8],[0.2,0.6]],
                                           model_classes=['Decision Tree', 
                                                          'Logistic Regression', 
                                                          'Random Forest', 
                                                          'XGBoost']):
    
    # Create as many graphs as there are performance metrics to display
    n_performance_metrics = len(performance_metrics_list)
    fig, ax = plt.subplots(1, n_performance_metrics, figsize=(5*n_performance_metrics,4))
    parameter_types=['Default parameters','Best validation parameters','Optimal parameters']
    # Plot performance metric for each metric in performance_metrics_list
    for i in range(n_performance_metrics):
        get_model_selection_performance_plot(performances_df_dictionary, 
                                             ax[i], 
                                             performance_metrics_list[i],
                                             ylim=ylim_list[i],
                                             model_classes=model_classes
                                            )
    ax[n_performance_metrics-1].legend(loc='upper left', 
                                       labels=parameter_types, 
                                       bbox_to_anchor=(1.05, 1),
                                       title="Parameter type",
                                       prop={'size': 12},
                                       title_fontsize=12)
    plt.subplots_adjust(wspace=0.5, hspace=0.8)
    
# ----------------------------------------------------
def plot_decision_boundary_classifier(ax, 
                                      classifier,
                                      train_df,
                                      input_features=['X1','X2'],
                                      output_feature='Y',
                                      title="",
                                      fs=14,
                                      plot_training_data=True):
    plot_colors = ["tab:blue","tab:orange"]
    x1_min, x1_max = train_df[input_features[0]].min() - 1, train_df[input_features[0]].max() + 1
    x2_min, x2_max = train_df[input_features[1]].min() - 1, train_df[input_features[1]].max() + 1
    plot_step=0.1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, plot_step), np.arange(x2_min, x2_max, plot_step))
    Z = classifier.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:,1]
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu_r,alpha=0.3)
    if plot_training_data:
        # Plot the training points
        groups = train_df.groupby(output_feature)
        for name, group in groups:
            ax.scatter(group[input_features[0]], group[input_features[1]], edgecolors='black', label=name)
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel(input_features[0], fontsize=fs)
    ax.set_ylabel(input_features[1], fontsize=fs)
    

# ----------------------------------------------------    
def kfold_cv_with_classifier(classifier,
                             X,
                             y,
                             n_splits=5,
                             strategy_name="Basline classifier"):
    
    cv = sklearn.model_selection.StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    cv_results_ = sklearn.model_selection.cross_validate(classifier,X,y,cv=cv,
                                                         scoring=['roc_auc',
                                                                  'average_precision',
                                                                  'balanced_accuracy'],
                                                         return_estimator=True)
    
    results = round(pd.DataFrame(cv_results_),3)
    results_mean = list(results.mean().values)
    results_std = list(results.std().values)
    results_df = pd.DataFrame([[str(round(results_mean[i],3))+'+/-'+
                                str(round(results_std[i],3)) for i in range(len(results))]],
                              columns=['Fit time (s)','Score time (s)',
                                       'AUC ROC','Average Precision','Balanced accuracy'])
    results_df.rename(index={0:strategy_name}, inplace=True)
    classifier_0 = cv_results_['estimator'][0]
    (train_index, test_index) = next(cv.split(X, y))
    train_df = pd.DataFrame({'X1':X[train_index,0], 'X2':X[train_index,1], 'Y':y[train_index]})
    test_df = pd.DataFrame({'X1':X[test_index,0], 'X2':X[test_index,1], 'Y':y[test_index]})
    return (results_df, classifier_0, train_df, test_df)


# ----------------------------------------------------
def plot_decision_boundary(classifier_0,
                           train_df, 
                           test_df):
    fig_decision_boundary, ax = plt.subplots(1, 3, figsize=(5*3,5))
    plot_decision_boundary_classifier(ax[0], classifier_0,
                                  train_df,
                                  title="Decision surface of the decision tree\n With training data",
                                  plot_training_data=True)
    plot_decision_boundary_classifier(ax[1], classifier_0,
                                  train_df,
                                  title="Decision surface of the decision tree\n",
                                  plot_training_data=False)
    plot_decision_boundary_classifier(ax[2], classifier_0,
                                  test_df,
                                  title="Decision surface of the decision tree\n With test data",
                                  plot_training_data=True)

    ax[-1].legend(loc='upper left', 
                  bbox_to_anchor=(1.05, 1),
                  title="Class")
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(vmin=0, vmax=1))
    cax = fig_decision_boundary.add_axes([0.93, 0.15, 0.02, 0.5])
    fig_decision_boundary.colorbar(sm, cax=cax, alpha=0.3, boundaries=np.linspace(0, 1, 11))
    return fig_decision_boundary