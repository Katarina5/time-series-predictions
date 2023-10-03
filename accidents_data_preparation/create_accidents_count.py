import pandas as pd
import glob
import re

# concat data from multiple CSVs to a single DataFrame
df_list = []
filepaths = ['./data-nehody/data_GIS_12-' + str(y) for y in range(2016, 2023)]
for path in filepaths:
    csv_files = glob.glob(path + "/*.csv")
    dfs_in_file = [pd.read_csv(file, sep=";", encoding='ANSI', header=None, low_memory=False) for file in csv_files]
    df_list.append(pd.concat(dfs_in_file, axis=0, ignore_index=True))

accidents_df = pd.concat(df_list, axis=0, ignore_index=True)
accidents_df['ds'] = pd.to_datetime(accidents_df[3])
accidents_df = accidents_df[['ds']]

# create one DataFrame with daily accident counts
daily_count = pd.DataFrame(accidents_df.groupby(accidents_df['ds'].dt.date)['ds'].count())

# add new dates (to be predicted) to the DataFrame
last_date = daily_count.index.max()
next_dates = pd.date_range(start=last_date, periods=14 + 1, freq='1d')
new_rows = pd.DataFrame(next_dates, columns=['ds'])
new_rows = new_rows.iloc[1:]
new_rows['ds'] = new_rows['ds'].dt.date  # extract only the date part (without time)
daily_count = pd.concat([daily_count, new_rows.set_index('ds')])

# add temperatures to the DataFrame
temperatures = pd.read_csv('temperatures_farenheit.csv', index_col=None)
temperatures['temperature'] = round((temperatures['temperature'] - 32) * 5 / 9, 1)
daily_count['temperature'] = temperatures['temperature'].values

# add hours of daylight to the DataFrame
daylight = pd.read_csv('daylight_hours.csv', index_col=None)

# function to convert the string from CSV to a number of hours
def convert_hour_string_to_decimal(time):
    parts = time.split(':')
    return (60 * 60 * int(parts[0]) + 60 * int(parts[1]) + int(parts[2])) / (60 * 60)

daylight['daylight'] = daylight['daylight'].apply(convert_hour_string_to_decimal)
daily_count['daylight'] = daylight['daylight'].values

daily_count.to_csv('accidents_daily.csv', header=['y', 'temp', 'daylight'])
