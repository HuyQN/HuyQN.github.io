#MAKE SURE TO CHANGE PATHS/RENAME DATA FILES
#
# Inputs: Yelp user+review data
# Output: user-score table
#
#mwu305@gatech.edu
import pandas as pd
import numpy as np

#params
length_factor = 0.2
weight_factor = 1
use_cap = 10
review_data = 'reviews.csv'
user_data = 'users.csv'

#IMPORT DATA
df = pd.read_csv('reviews.csv')
df = df[['user_id','text','date']]

df_users = pd.read_csv('users.csv')
df_users = df_users[['user_id','name','review_count','yelping_since','useful']]

#FILTER USERS WITH NO REVIEWS
df_users = df_users[df_users['review_count'] > 0]
#GET RATIO OF USEFUL REACTIONS TO NUMBER OF REVIEWS -- MAIN ADJUSTMENT/WEIGHT
#0.2% OF USERS HAVE SCORES ABOVE 10
df_users['use_ratio'] = (df_users['useful']/df_users['review_count']).clip(0,use_cap)
df['text_length'] = df['text'].str.len()


#AGGREGATE AND FIND TEXT LENGTH 
#LOG NORMAL DISTRIBUTION W/ OUTLIERS REMOVED SO MOST VALUES SHOULD HAVE THE SAME MULTIPLIER
#MAINLY A PENALTY FOR VERY SHORT REVIEWS. ADJUST WITH LENGTH_FACTOR ABOVE
df_length = df.groupby('user_id').mean('text_length').clip(0,1000).reset_index()
df_length['text_length'] = df_length['text_length'].apply(np.sqrt)

df_all = df_users.merge(df_length,on='user_id',how='right')
df_all.text_length -= df_all.text_length.min()
df_all.text_length /= df_all.text_length.max()
df_all['text_length'] = (df_all.text_length*length_factor)+1

#FIND REVIEW VOLUME AND GIVE SMALL BONUS/PENALTY 
#DAMPEN LOW VOLUME HIGH-SCORES AND BOOST ACTIVE REVIEWERS
df_all['count_coef'] = df_all['review_count'].apply(np.log)
df_all.count_coef -= df_all.count_coef.min()
df_all.count_coef /= df_all.count_coef.max()
df_all.count_coef = df_all.count_coef.clip(0,0.7)
df_all.count_coef = df_all.count_coef-df_all.count_coef.mean()+1

#CREDIBILITY = USEFULNESS * VOLUME ADJUSTMENT * QUALITY BONUS
df_all['credibility'] = ((df_all['use_ratio']+1) * df_all.count_coef * df_all.text_length) ** weight_factor

#EXPORT DATA
df_all[['user_id','credibility']].to_csv('user_postprocess.csv')