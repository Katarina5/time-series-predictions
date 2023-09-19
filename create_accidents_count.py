import pandas as pd
import glob

# concat data from multiple CSVs to a single dataframe
df_list = []
filepaths = ['./data-nehody/data_GIS_12-' + str(y) for y in range(2016, 2023)]
for path in filepaths:
    csv_files = glob.glob(path + "/*.csv")
    dfs_in_file = [pd.read_csv(file, sep=";", encoding='ANSI', header=None, low_memory=False) for file in csv_files]
    df_list.append(pd.concat(dfs_in_file, axis=0, ignore_index=True))

accidents_df = pd.concat(df_list, axis=0, ignore_index=True)
accidents_df['date'] = pd.to_datetime(accidents_df[3])
accidents_df['weekday'] = accidents_df[4]
accidents_df = accidents_df[['date', 'weekday']]

daily_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.date)['date'].count())
monthly_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count())

daily_count.to_csv('accidents_daily.csv', header=['y'])
monthly_count.to_csv('accidents_monthly.csv', header=['y'])

temperatures = pd.read_csv('temperatures_farenheit.csv', index_col=None)
temperatures['temperature'] = round((temperatures['temperature'] - 32) * 5 / 9, 1)
daily_count['temperature'] = temperatures['temperature'].values
daily_count.to_csv('accidents_daily_temp.csv', header=['y', 'temp'])
