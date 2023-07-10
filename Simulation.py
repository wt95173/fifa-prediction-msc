import pandas as pd
from joblib import load

rf_model = load('../models/rf_model.joblib')
df_base = pd.read_csv('../dataset/results2/df_base_new.csv')

wc_group = {'A': [['Qatar', 0], ['Ecuador', 0], ['Senegal', 0], ['Netherlands', 0]],
            'B': [['England', 0], ['Iran', 0], ['United States', 0], ['Wales', 0]],
            'C': [['Argentina', 0], ['Saudi Arabia', 0], ['Mexico', 0], ['Poland', 0]],
            'D': [['France', 0], ['Australia', 0], ['Denmark', 0], ['Tunisia', 0]],
            'E': [['Spain', 0], ['Costa Rica', 0], ['Germany', 0], ['Japan', 0]],
            'F': [['Belgium', 0], ['Canada', 0], ['Morocco', 0], ['Croatia', 0]],
            'G': [['Brazil', 0], ['Serbia', 0], ['Switzerland', 0], ['Cameroon', 0]],
            'H': [['Portugal', 0], ['Ghana', 0], ['Uruguay', 0], ['South Korea', 0]]}
print(wc_group['A'][0][0])
features = df_base[df_base.columns[3:19].values]
