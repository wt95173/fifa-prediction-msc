import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
import seaborn as sns

df_base = pd.read_csv('../dataset/results/df_base_no.csv')

df_base["target"] = df_base["result"].apply(lambda x: 1 if x == 2 else x)

df_base.to_csv('../dataset/results2/df_base.csv', index=False)

# Find key features
features = df_base[df_base.columns[8:40].values]
target = df_base[df_base.columns[40:]]
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
     'home_player_dif_mean', 'home_player_dif_mean_l5', 'is_friendly', 'target']]

df_base_new_copy = df_base_new.copy()

# Create some new features
df_base_new_copy["goals_dif"] = df_base["home_goals_mean"] - df_base["away_goals_mean"]
df_base_new_copy["goals_l5_dif"] = df_base["home_goals_mean_l5"] - df_base["away_goals_mean_l5"]
df_base_new_copy["goals_ano_dif"] = df_base["home_goals_mean_ano"] - df_base["away_goals_mean_ano"]
df_base_new_copy["goals_l5_ano_dif"] = df_base["home_goals_mean_l5_ano"] - df_base["away_goals_mean_l5_ano"]
df_base_new_copy["rank_mean_dif"] = df_base["home_rank_mean"] - df_base["away_rank_mean"]
df_base_new_copy["rank_mean_l5_dif"] = df_base["home_rank_mean_l5"] - df_base["away_rank_mean_l5"]
df_base_new_copy["points_mean_dif"] = df_base["home_points_mean"] - df_base["away_points_mean"]
df_base_new_copy["points_mean_l5_dif"] = df_base["home_points_mean_l5"] - df_base["away_points_mean_l5"]
df_base_new_copy["game_points_dif"] = df_base["home_game_points_mean"] - df_base["away_game_points_mean"]
df_base_new_copy["game_points_l5_dif"] = df_base["home_game_points_mean_l5"] - df_base["away_game_points_mean_l5"]
df_base_new_copy["game_points_rank_dif"] = df_base["home_game_points_rank_mean"] - df_base["away_game_points_rank_mean"]
df_base_new_copy["game_points_rank_l5_dif"] = df_base["home_game_points_rank_mean_l5"] - df_base[
    "away_game_points_rank_mean_l5"]
df_base_new_copy["player_dif_mean_dif"] = df_base["home_player_dif_mean"] - df_base["away_player_dif_mean"]
df_base_new_copy["player_dif_mean_l5_dif"] = df_base["home_player_dif_mean_l5"] - df_base["away_player_dif_mean_l5"]

# Find key features in new features
df_base_new_copy = df_base_new_copy[
    ["date", "home_team", "away_team", 'rank_dif', "goals_dif", "goals_l5_dif", "goals_ano_dif", "goals_l5_ano_dif",
     "rank_mean_dif", "rank_mean_l5_dif", "points_mean_dif", "points_mean_l5_dif", "game_points_dif",
     "game_points_l5_dif", "game_points_rank_dif", "game_points_rank_l5_dif", "player_dif_mean_dif",
     "player_dif_mean_l5_dif", 'is_friendly', 'target']]

features2 = df_base_new_copy[df_base_new_copy.columns[4:19].values]
target2 = df_base_new_copy[df_base_new_copy.columns[19:]]
target2 = np.ravel(target2)

features_new2 = selector.fit_transform(features2, target2)

selected_features2 = features2.columns[selector.get_support()]

print(selected_features2)

# Determine the final data to be included
df_base_new_copy = df_base_new_copy[
    ["date", "home_team", "away_team", 'rank_dif', 'goals_dif', 'goals_l5_dif', 'goals_ano_dif', 'goals_l5_ano_dif',
     'rank_mean_dif', 'rank_mean_l5_dif', 'points_mean_dif', "points_mean_l5_dif", 'game_points_dif',
     'game_points_l5_dif',
     'game_points_rank_dif', "player_dif_mean_dif", "player_dif_mean_l5_dif", 'is_friendly', 'target']]

# Graphing relationships
df_base_draw = df_base_new_copy.copy()

cols = ['rank_dif', 'goals_dif', 'goals_l5_dif', 'goals_ano_dif', 'goals_l5_ano_dif', 'rank_mean_dif',
        'rank_mean_l5_dif', 'points_mean_dif', "points_mean_l5_dif", 'game_points_dif', 'game_points_l5_dif',
        'game_points_rank_dif', "player_dif_mean_dif", "player_dif_mean_l5_dif"]  # no 'is_friendly'

df_base_draw['target'] = df_base_draw['target'].astype('category')

plt.figure(figsize=(15, 10))
correlation_matrix = df_base_draw[cols + ['target']].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.xticks(rotation=45)
plt.savefig(f'../images/correlation_matrix_full.png')
plt.show()

# Violin diagram
# for feature in cols:
#     plt.figure(figsize=(10, 6))
#     sns.violinplot(x='target', y=feature, data=df_base_draw, split=True)
#     plt.title(f'{feature} by target category')
#     plt.savefig(f'../images/violinplot_{feature}.png')
#     plt.show()

# violin_data = df_base_new_copy.melt(id_vars='target', value_vars=cols, var_name='features', value_name='value')
# plt.figure(figsize=(12, 10))
# sns.violinplot(x="features", y="value", hue="target", data=violin_data, split=True, inner="quart")
# plt.xticks(rotation=45)
# plt.savefig(f'../images/violinplot_full.png')
# plt.show()

for feature in cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y=feature, data=df_base_draw)
    plt.title(f'{feature} by target category')
    plt.savefig(f'../images/boxplot_{feature}.png')
    plt.show()

box_data = df_base_new_copy.melt(id_vars='target', value_vars=cols, var_name='features', value_name='value')
plt.figure(figsize=(12, 10))
sns.boxplot(x="features", y="value", hue="target", data=box_data)
plt.xticks(rotation=45)
plt.savefig(f'../images/boxplot_full.png')
plt.show()

cols2 = ['goals_dif', 'goals_l5_dif', 'goals_ano_dif', 'goals_l5_ano_dif', 'game_points_dif', 'game_points_l5_dif']
box_data2 = df_base_new_copy.melt(id_vars='target', value_vars=cols2, var_name='features', value_name='value')
plt.figure(figsize=(12, 10))
sns.boxplot(x="features", y="value", hue="target", data=box_data2)
plt.xticks(rotation=45)
plt.savefig(f'../images/boxplot_full2.png')
plt.show()

sns.scatterplot(data=df_base_draw, x='goals_dif', y='goals_l5_dif', hue='target')
plt.savefig('../images/scatterplot_goals_dif_goals_l5_dif.png')
plt.show()

sns.scatterplot(data=df_base_draw, x='rank_mean_dif', y='rank_mean_l5_dif', hue='target')
plt.savefig('../images/scatterplot_rank_mean_dif_rank_mean_l5_dif.png')
plt.show()

sns.scatterplot(data=df_base_draw, x='goals_dif', y='goals_ano_dif', hue='target')
plt.savefig('../images/scatterplot_goals_dif_goals_ano_dif.png')
plt.show()

sns.scatterplot(data=df_base_draw, x='rank_dif', y='points_mean_dif', hue='target')
plt.savefig('../images/scatterplot_rank_dif_points_mean_dif.png')
plt.show()

# Saving of processed data
df_base_new_final = df_base_new_copy[
    ["date", "home_team", "away_team", 'rank_dif', 'goals_dif', 'goals_ano_dif', 'goals_l5_ano_dif',
     'rank_mean_dif', 'points_mean_dif', "points_mean_l5_dif", 'game_points_dif',
     'game_points_l5_dif', 'game_points_rank_dif', "player_dif_mean_dif", "player_dif_mean_l5_dif", 'is_friendly',
     'target']]
df_base_new_final.to_csv('../dataset/results2/df_base_new.csv', index=False)
