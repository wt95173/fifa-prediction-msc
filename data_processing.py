import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

# Read match data
df_results = pd.read_csv("../dataset/FIFA_MATCH/results.csv")
df_results["date"] = pd.to_datetime(df_results["date"])

# Selecting the forecast interval
df_results_2018 = df_results[(df_results["date"] >= "2018-7-22") & (df_results["date"] < "2022-11-20")].reset_index(
    drop=True)

# Read ranking data
df_rank = pd.read_csv("../dataset/fifa_ranking-2022-12-22.csv")
df_rank["rank_date"] = pd.to_datetime(df_rank["rank_date"])
df_rank_2018 = df_rank[(df_rank["rank_date"] >= "2018-7-22") & (df_rank["rank_date"] < "2022-11-20")].reset_index(
    drop=True)

# Country name change
replacement_dict = {
    "IR Iran": "Iran",
    "Korea DPR": "North Korea",
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Bosnia Herzegovina": "Bosnia and Herzegovina"
}
df_rank_2018["country_full"] = df_rank_2018["country_full"].replace(replacement_dict)
df_rank_2018["country_full"] = df_rank_2018["country_full"].str.replace("&", "and")

# Combined rankings and specific matches
df_rank_2018 = df_rank_2018.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample(
    'D').first().ffill().reset_index()

df_rank_wc = df_results_2018.merge(
    df_rank_2018[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]],
    left_on=["date", "home_team"], right_on=["rank_date", "country_full"]).drop(["rank_date", "country_full"], axis=1)

df_rank_wc = df_rank_wc.merge(
    df_rank_2018[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]],
    left_on=["date", "away_team"], right_on=["rank_date", "country_full"], suffixes=("_home", "_away")).drop(
    ["rank_date", "country_full"], axis=1)

# Merging player data

df_player_18 = pd.read_csv("../dataset/FIFA_PLAYER/FIFA18_official_data.csv")
df_player_19 = pd.read_csv("../dataset/FIFA_PLAYER/FIFA19_official_data.csv")
df_player_20 = pd.read_csv("../dataset/FIFA_PLAYER/FIFA20_official_data.csv")
df_player_21 = pd.read_csv("../dataset/FIFA_PLAYER/FIFA21_official_data.csv")
df_player_22 = pd.read_csv("../dataset/FIFA_PLAYER/FIFA22_official_data.csv")

start_dates = ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01"]
end_dates = ["2018-12-31", "2019-12-31", "2020-12-31", "2021-12-31", "2022-12-31"]
df_player_list = [df_player_18, df_player_19, df_player_20, df_player_21, df_player_22]
df_player_final = pd.DataFrame()

for df_player, start_date, end_date in zip(df_player_list, start_dates, end_dates):
    df_player_top = df_player.groupby("Nationality").apply(lambda x: x.nlargest(18, "Overall")).reset_index(drop=True)
    df_player_top["Nationality"] = df_player_top["Nationality"].replace(replacement_dict)
    df_player_top["Nationality"] = df_player_top["Nationality"].str.replace("&", "and")
    df_player_avg = df_player_top.groupby("Nationality")["Overall"].agg(["mean"]).reset_index()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dfs = []
    for date in dates:
        df_temp = df_player_avg.copy()
        df_temp['date_a'] = date
        dfs.append(df_temp)
    df_player_a = pd.concat(dfs, ignore_index=True)
    df_player_a["date_a"] = pd.to_datetime(df_player_a["date_a"])
    df_player_final = pd.concat([df_player_final, df_player_a], ignore_index=True)

df_player_final = df_player_final.rename(columns={'mean': 'player_mean'})

df_rank_wc = pd.merge(df_rank_wc, df_player_final[["date_a", "Nationality", "player_mean"]],
                      left_on=["date", "home_team"],
                      right_on=["date_a", "Nationality"], how="left")
df_rank_wc = df_rank_wc.drop(["date_a", "Nationality"], axis=1)

df_rank_wc = pd.merge(df_rank_wc, df_player_final[["date_a", "Nationality", "player_mean"]],
                      left_on=["date", "away_team"],
                      right_on=["date_a", "Nationality"], how="left", suffixes=("_home", "_away"))
df_rank_wc = df_rank_wc.drop(["date_a", "Nationality"], axis=1)
df_rank_wc.to_csv('../dataset/results2/df_rank_wc2.csv', index=False)

# Setting the results of the competition
df_compare = df_rank_wc


def result_finder(home, away):
    if home > away:
        return pd.Series([0, 3, 0])
    if home < away:
        return pd.Series([1, 0, 3])
    else:
        return pd.Series([2, 1, 1])


results = df_compare.apply(lambda x: result_finder(x["home_score"], x["away_score"]), axis=1)
df_compare[["result", "home_team_points", "away_team_points"]] = results

df_compare["rank_dif"] = df_compare["rank_home"] - df_compare["rank_away"]
df_compare["player_mean_dif"] = df_compare["player_mean_home"] - df_compare["player_mean_away"]
df_compare["points_home_by_rank"] = df_compare["home_team_points"] / df_compare["rank_away"]
df_compare["points_away_by_rank"] = df_compare["away_team_points"] / df_compare["rank_home"]

df_home = df_compare[
    ["date", "home_team", "home_score", "away_score", "rank_home", "rank_away", "rank_change_home", "total_points_home",
     "result", "rank_dif", "points_home_by_rank", "home_team_points", "player_mean_dif"]]

df_away = df_compare[
    ["date", "away_team", "away_score", "home_score", "rank_away", "rank_home", "rank_change_away", "total_points_away",
     "result", "rank_dif", "points_away_by_rank", "away_team_points", "player_mean_dif"]]

# Rename column
column_mapping_away = {
    'away_score': 'score_ano',
    'rank_away': 'rank_ano'
}
df_home = df_home.rename(columns=column_mapping_away)
df_home.columns = [h.replace("home_", "").replace("_home", "") for h in df_home.columns]

column_mapping_home = {
    'home_score': 'score_ano',
    'rank_home': 'rank_ano'
}
df_away = df_away.rename(columns=column_mapping_home)
df_away.columns = [a.replace("away_", "").replace("_away", "") for a in df_away.columns]

# Combine home and away team results
df_team_stats = pd.concat([df_home, df_away])
df_team_stats.to_csv('../dataset/results/df_team_stats.csv', index=False)

# Extraction of more features
stats_val = []

for index, row in df_team_stats.iterrows():
    team = row["team"]
    date = row["date"]
    past_games = df_team_stats.query('team == @team and date < @date').sort_values(by='date', ascending=False)
    last5 = past_games.head(5)

    goals = past_games["score"].mean()
    goals_l5 = last5["score"].mean()

    goals_ano = past_games["score_ano"].mean()
    goals_ano_l5 = last5["score_ano"].mean()

    rank = past_games["rank_ano"].mean()
    rank_l5 = last5["rank_ano"].mean()

    if len(last5) > 0:
        points = past_games["total_points"].values[0] - past_games["total_points"].values[-1]
        points_l5 = last5["total_points"].values[0] - last5["total_points"].values[-1]
    else:
        points = 0
        points_l5 = 0

    gp = past_games["team_points"].mean()
    gp_l5 = last5["team_points"].mean()

    gp_rank = past_games["points_by_rank"].mean()
    gp_rank_l5 = last5["points_by_rank"].mean()

    pm = past_games["player_mean_dif"].mean()
    pm_l5 = last5["player_mean_dif"].mean()

    stats_val.append(
        [goals, goals_l5, goals_ano, goals_ano_l5, rank, rank_l5, points, points_l5, gp, gp_l5, gp_rank, gp_rank_l5, pm,
         pm_l5])

stats_cols = ["goals_mean", "goals_mean_l5", "goals_mean_ano", "goals_mean_l5_ano", "rank_mean", "rank_mean_l5",
              "points_mean", "points_mean_l5", "game_points_mean", "game_points_mean_l5", "game_points_rank_mean",
              "game_points_rank_mean_l5", "player_dif_mean", "player_dif_mean_l5"]

df_stats_cols = pd.DataFrame(stats_val, columns=stats_cols)

df_full = pd.concat([df_team_stats.reset_index(drop=True), df_stats_cols], axis=1, ignore_index=False)

# Split and combined into a final table
df_home_team_stats = df_full.iloc[:int(df_full.shape[0] / 2), :]
df_away_team_stats = df_full.iloc[int(df_full.shape[0] / 2):, :]

df_home_team_stats = df_home_team_stats[df_home_team_stats.columns[-14:]]
df_away_team_stats = df_away_team_stats[df_away_team_stats.columns[-14:]]

df_home_team_stats.columns = ['home_' + str(col) for col in df_home_team_stats.columns]
df_away_team_stats.columns = ['away_' + str(col) for col in df_away_team_stats.columns]

df_match_stats = pd.concat([df_home_team_stats, df_away_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)

df_full = pd.concat([df_compare, df_match_stats.reset_index(drop=True)], axis=1, ignore_index=False)

df_full['is_friendly_0'] = np.where(df_full['tournament'] == 'Friendly', 1, 0)
df_full['is_friendly_1'] = np.where(df_full['tournament'] == 'Friendly', 0, 1)

df_full.to_csv('../dataset/results/df_full.csv', index=False)

df_base = df_full[
    ["date", "home_team", "away_team", "rank_home", "rank_away", "home_score", "away_score", "result", "rank_dif",
     "rank_change_home", "rank_change_away", 'home_goals_mean', 'home_goals_mean_l5', 'home_goals_mean_ano',
     'home_goals_mean_l5_ano', 'home_rank_mean', 'home_rank_mean_l5', 'home_points_mean', 'home_points_mean_l5',
     'away_goals_mean', 'away_goals_mean_l5', 'away_goals_mean_ano', 'away_goals_mean_l5_ano', 'away_rank_mean',
     'away_rank_mean_l5', 'away_points_mean', 'away_points_mean_l5', 'home_game_points_mean',
     'home_game_points_mean_l5', 'home_game_points_rank_mean', 'home_game_points_rank_mean_l5', 'away_game_points_mean',
     'away_game_points_mean_l5', 'away_game_points_rank_mean', 'away_game_points_rank_mean_l5', 'away_player_dif_mean',
     'away_player_dif_mean_l5', 'home_player_dif_mean', 'home_player_dif_mean_l5', 'is_friendly_0', 'is_friendly_1']]

df_base_no = df_base.dropna()

df_base_no.to_csv('../dataset/results/df_base_no.csv', index=False)
