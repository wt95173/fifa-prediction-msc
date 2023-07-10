import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns

df_base = pd.read_csv('../dataset/results/df_base_no.csv')

df_base["target"] = df_base["result"].apply(lambda x: 1 if x == 2 else x)

df_base.to_csv('../dataset/results2/df_base.csv', index=False)

# Find key features
features = df_base[df_base.columns[8:41].values]
target = df_base[df_base.columns[41:]]
target = np.ravel(target)

k = 10
selector = SelectKBest(score_func=f_regression, k=k)
features_new = selector.fit_transform(features, target)

selected_features = features.columns[selector.get_support()]

print(selected_features)

df_base_new = df_base[
    ["date", "home_team", "away_team", 'rank_dif', 'home_goals_mean', 'home_goals_mean_ano', 'home_goals_mean_l5_ano',
     'away_goals_mean', 'away_goals_mean_ano', 'home_game_points_mean', 'home_game_points_mean_l5',
     'away_game_points_mean', 'away_game_points_mean_l5', 'away_player_dif_mean', 'away_player_dif_mean_l5',
     'home_player_dif_mean', 'home_player_dif_mean_l5', 'is_friendly_0', 'is_friendly_1', 'target']]

# Create some new features
df_base_new["goals_dif"] = df_base["home_goals_mean"] - df_base["away_goals_mean"]
df_base_new["goals_l5_dif"] = df_base["home_goals_mean_l5"] - df_base["away_goals_mean_l5"]
df_base_new["goals_ano_dif"] = df_base["home_goals_mean_ano"] - df_base["away_goals_mean_ano"]
df_base_new["goals_l5_ano_dif"] = df_base["home_goals_mean_l5_ano"] - df_base["away_goals_mean_l5_ano"]
df_base_new["rank_mean_dif"] = df_base["home_rank_mean"] - df_base["away_rank_mean"]
df_base_new["rank_mean_l5_dif"] = df_base["home_rank_mean_l5"] - df_base["away_rank_mean_l5"]
df_base_new["points_mean_dif"] = df_base["home_points_mean"] - df_base["away_points_mean"]
df_base_new["points_mean_l5_dif"] = df_base["home_points_mean_l5"] - df_base["away_points_mean_l5"]
df_base_new["game_points_dif"] = df_base["home_game_points_mean"] - df_base["away_game_points_mean"]
df_base_new["game_points_l5_dif"] = df_base["home_game_points_mean_l5"] - df_base["away_game_points_mean_l5"]
df_base_new["game_points_rank_dif"] = df_base["home_game_points_rank_mean"] - df_base["away_game_points_rank_mean"]
df_base_new["game_points_rank_l5_dif"] = df_base["home_game_points_rank_mean_l5"] - df_base[
    "away_game_points_rank_mean_l5"]
df_base_new["player_dif_mean_dif"] = df_base["home_player_dif_mean"] - df_base["away_player_dif_mean"]
df_base_new["player_dif_mean_l5_dif"] = df_base["home_player_dif_mean_l5"] - df_base["away_player_dif_mean_l5"]

# Find key features in new features
df_base_new = df_base_new[
    ["date", "home_team", "away_team", 'rank_dif', "goals_dif", "goals_l5_dif", "goals_ano_dif", "goals_l5_ano_dif",
     "rank_mean_dif", "rank_mean_l5_dif", "points_mean_dif", "points_mean_l5_dif", "game_points_dif",
     "game_points_l5_dif", "game_points_rank_dif", "game_points_rank_l5_dif", "player_dif_mean_dif",
     "player_dif_mean_l5_dif", 'is_friendly_0', 'is_friendly_1', 'target']]

features2 = df_base_new[df_base_new.columns[4:20].values]
target2 = df_base_new[df_base_new.columns[20:]]
target2 = np.ravel(target2)

features_new2 = selector.fit_transform(features2, target2)

selected_features2 = features2.columns[selector.get_support()]

print(selected_features2)

df_base_new = df_base_new[
    "date", "home_team", "away_team", 'rank_dif', 'goals_dif', 'goals_l5_dif', 'goals_ano_dif', 'goals_l5_ano_dif',
    'rank_mean_dif', 'rank_mean_l5_dif', 'points_mean_dif', 'game_points_dif', 'game_points_l5_dif',
    'game_points_rank_dif', "player_dif_mean_dif", "player_dif_mean_l5_dif", 'is_friendly_0', 'is_friendly_1', 'target']

cols = ['rank_dif', 'goals_dif', 'goals_l5_dif', 'goals_ano_dif', 'goals_l5_ano_dif', 'rank_mean_dif',
        'rank_mean_l5_dif', 'points_mean_dif', 'game_points_dif', 'game_points_l5_dif', 'game_points_rank_dif',
        "player_dif_mean_dif", "player_dif_mean_l5_dif", 'is_friendly_0', 'is_friendly_1']

sns.pairplot(df_base_new[cols], hue="target", markers=["o", "s"])
plt.savefig("../pairplot.png")
plt.show()

df_base_new.to_csv('../dataset/results2/df_base_new.csv', index=False)
