import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
sns.set_style('darkgrid', {'axes.facecolor': '1'})

# ----------------------------------------------------------------------
def Simulate_customer_profile_df(n_customers, random_state=66):
    np.random.seed(random_state)
    customer_id_property=[]
    # Generate customer location from uniform distributions 
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0,100)
        y_customer_id = np.random.uniform(0,100)
        mean_amount = np.random.uniform(10,100) 
        std_amount = mean_amount/2.5 
        mean_trans_per_day = np.random.uniform(0,5) # mean transactions nearby per day 
        customer_id_property.append([customer_id,
                                      x_customer_id, y_customer_id,
                                      mean_amount, std_amount,
                                      mean_trans_per_day])
    customer_profiles_df = pd.DataFrame(customer_id_property, columns=['CUSTOMER_ID',
                                                                      'location_x', 'location_y',
                                                                      'Mean_trans', 'STD_trans',
                                                                      'num_tran_day'])
    return customer_profiles_df

# ----------------------------------------------------------------------
def Simulate_station_profiles_df(n_stations, random_state=66):
    np.random.seed(random_state)
    station_id_property=[]
    # Generate station locations from uniform distributions 
    for store_id in range(n_stations):
        x_station_id = np.random.uniform(0,100)
        y_station_id = np.random.uniform(0,100)
        station_id_property.append([store_id,
                                      x_station_id, y_station_id])                         
    station_profiles_df = pd.DataFrame(station_id_property, columns=['STORE_ID',
                                                                      'Station_location_x', 'Station_location_y'])
    return station_profiles_df

# ----------------------------------------------------------------------
def station_within_radius(customer_profile,x_y_stations, r):
    x_y_customer = customer_profile[['location_x','location_y']].values.astype(float)
    # Root of sum of squared difference
    squared_diff = np.square(x_y_customer - x_y_stations)
    # compute suared root to get distance
    dist = np.sqrt(np.sum(squared_diff, axis=1))
    # Get the indices of terminals which are at a distance within r
    available_stations = list(np.where(dist<r)[0])
    # Return the list of stations
    return available_stations

# ----------------------------------------------------------------------
def Simulate_transactions_df(customer_profile, start_date = "2022-01-01", nb_days = 50):
    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
    # For all days
    for day in range(nb_days):
        # Generate number of transactions from poisson distribution
        nb_trans = np.random.poisson(customer_profile.num_tran_day)
        # Start generating transactions
        if nb_trans>0:
            for trans in range(nb_trans):
                # We are trying to assume that most transactions happen during daytime
                time_trans = int(np.random.normal(86400/2, 20000))
                # We have at most 24 hours for a single day
                if (time_trans>0) and (time_trans<86400):
                    # Transaction amount is drawn from a normal distribution  
                    amount = np.random.normal(customer_profile.Mean_trans,customer_profile.STD_trans)
                    # If amount negative, we resample from a uniform distribution
                    if amount<0:
                        amount = np.random.uniform(0,customer_profile.Mean_trans*2)
                    amount=np.round(amount,decimals=2)
                    #Transactions need to happen at available stations.
                    if len(customer_profile.available_stations)>0:
                        store_id = random.choice(customer_profile.available_stations)
                        customer_transactions.append([time_trans+day*86400, day,customer_profile.CUSTOMER_ID, store_id, amount])
            #We combine the all of the columns together.
    customer_transactions = pd.DataFrame(customer_transactions, columns=['Trans_TIME_SECONDS', 'Trans_TIME_DAYS', 'CUSTOMER_ID', 'STORE_ID', 'Trans_AMOUNT'])
    if len(customer_transactions)>0:
        customer_transactions['Trans_DATETIME'] = pd.to_datetime(customer_transactions["Trans_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions=customer_transactions[['Trans_DATETIME','CUSTOMER_ID', 'STORE_ID', 'Trans_AMOUNT','Trans_TIME_SECONDS', 'Trans_TIME_DAYS']]
    return customer_transactions  

# ----------------------------------------------------------------------
#Go through the previous process all together.
def Simulate_dataset(n_customers = 10000, n_stations = 1000000, nb_days=90, start_date="2022-01-01", r=7):
    customer_profiles_df = Simulate_customer_profile_df(n_customers, random_state = 3)
    station_profiles_df = Simulate_station_profiles_df(n_stations, random_state = 33)
    x_y_stations = station_profiles_df[['Station_location_x','Station_location_y']].values.astype(float)
    customer_profiles_df['available_stations'] = customer_profiles_df.apply(lambda x : station_within_radius(x, x_y_stations=x_y_stations, r=r), axis=1)
    customer_profiles_df['nb_stations']=customer_profiles_df.available_stations.apply(len)
    transactions_df=customer_profiles_df.groupby('CUSTOMER_ID').apply(lambda x : Simulate_transactions_df(x.iloc[0], nb_days=nb_days)).reset_index(drop=True)
    transactions_df=transactions_df.sort_values('Trans_DATETIME')
    transactions_df.reset_index(inplace=True,drop=True)
    transactions_df.reset_index(inplace=True)
    transactions_df.rename(columns = {'index':'TRANSACTION_ID'}, inplace = True)
    return (customer_profiles_df, station_profiles_df, transactions_df)

# ----------------------------------------------------------------------
def Simulate_frauds(customer_profiles_df, station_profiles_df, transactions_df):
    transactions_df['Trans_FRAUD']=0
    #Transaction amount greater than 190 is considered fraud.
    transactions_df.loc[transactions_df.Trans_AMOUNT>190, 'Trans_FRAUD']=1
    #9 stations are frauds.
    for day in range(transactions_df.Trans_TIME_DAYS.max()):
        fraud_stations = station_profiles_df.STORE_ID.sample(n=9, random_state=day)
        fraud_transactions=transactions_df[(transactions_df.Trans_TIME_DAYS>=day) & (transactions_df.Trans_TIME_DAYS<day+10) &(transactions_df.STORE_ID.isin(fraud_stations))]
        transactions_df.loc[fraud_transactions.index,'Trans_FRAUD']=1
    #10 customers are selected as fraud victims.
    for day in range(transactions_df.Trans_TIME_DAYS.max()):
        fraud_customers = customer_profiles_df.CUSTOMER_ID.sample(n=10, random_state=day).values
        fraud_transactions=transactions_df[(transactions_df.Trans_TIME_DAYS>=day) & (transactions_df.Trans_TIME_DAYS<day+7) & (transactions_df.CUSTOMER_ID.isin(fraud_customers))]
        num_fraud_transactions=len(fraud_transactions)
        random.seed(day)
        index = random.sample(list(fraud_transactions.index.values),k=int(num_fraud_transactions/3))
        #We modify their transaction values hoping to be mimic real life scenarios.
        transactions_df.loc[index,'Trans_FRAUD']=transactions_df.loc[index,'Trans_AMOUNT']*3
        transactions_df.loc[index,'Trans_FRAUD']=1        
    return transactions_df  