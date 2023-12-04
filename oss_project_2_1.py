import pandas as pd

file_path = "2019_kbo_for_kaggle_v2.csv"
data = pd.read_csv(file_path)

# Task 1: Print the top 10 players in hits, batting average, homerun, and on-base percentage for each year from 2015 to 2018.
for year in range(2015, 2019):
    year_data = data[data['year'] == year]
    print(f"\nTop 10 players in {year}:")

    # Hits
    top_hits = year_data.nlargest(10, 'H')[['batter_name', 'H']]
    print(f"Top 10 in hits:\n{top_hits}")

    # Batting average
    top_avg = year_data.nlargest(10, 'avg')[['batter_name', 'avg']]
    print(f"Top 10 in batting average:\n{top_avg}")

    # Homerun
    top_hr = year_data.nlargest(10, 'HR')[['batter_name', 'HR']]
    print(f"Top 10 in homerun:\n{top_hr}")

    # On-base percentage
    top_obp = year_data.nlargest(10, 'OBP')[['batter_name', 'OBP']]
    print(f"Top 10 in on-base percentage:\n{top_obp}")

# Task 2: Print the player with the highest war by position (cp) in 2018.
year_2018_data = data[data['year'] == 2018]
highest_war_by_position = year_2018_data.groupby('cp')['war'].idxmax()
top_war_players_by_position = year_2018_data.loc[highest_war_by_position, ['cp', 'batter_name', 'war']]
print("\nPlayer with the highest war by position in 2018:\n", top_war_players_by_position)

# Task 3: Calculate the indicator with the highest correlation with salary.
features = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
correlations = data[features + ['salary']].corr()['salary'].drop('salary')  # Exclude salary's own correlation
highest_correlation_feature = correlations.idxmax()
print("\nThe indicator with the highest correlation with salary is:", highest_correlation_feature)
