import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


sns.set_theme(style="white")
sns.set(rc={"figure.figsize": (8, 4), "figure.dpi": 300})

file_path_demand = '../datasets/dhn_demand.csv'
file_path_weather = '../datasets/weather_denmark.csv'


def load_and_merge_data(dhn_path, weather_path):
    df_dhn = pd.read_csv(dhn_path, parse_dates=[0]).set_index('time_rounded')
    df_weather = pd.read_csv(
        weather_path,
        header=0,
        names=['time_rounded', 'Temp', 'solar', 'wind', 'humid'],
        parse_dates=[0]
    ).set_index('time_rounded')

    df = df_dhn.join(df_weather, how='inner')

    print('Duplicated values:')
    print(df.duplicated().sum())
    print('Nun values:')
    print(df.isnull().sum())
    print('Index is increasing?')
    print(df.index.is_monotonic_increasing)

    df = df[~df.duplicated()]
    df = df.dropna()
    return df

def add_time_features(df):
    df['dow'] = df.index.dayofweek
    df['doy'] = df.index.dayofyear
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['hour'] = df.index.hour
    df['weekday'] = df.index.day_name()
    df['woy'] = df.index.isocalendar().week
    df['dom'] = df.index.day
    df['date'] = df.index.date
    return df

def add_holiday_weekend_features(df, country_code='DK'):
    holiday_dates = holidays.CountryHoliday(country_code)
    df['Holidays'] = df.index.to_series().apply(lambda x: x in holiday_dates).astype(int)
    df['weekend'] = df['dow'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def scale_demand(df):
    scaler = MinMaxScaler()
    df['Demand_scaled'] = scaler.fit_transform(df['Demand'].values.reshape(-1, 1))
    return df, scaler

def generate_cyclical_features(df, col_name):
    period = len(df[col_name].unique())
    df[f'sin_{col_name}'] = np.sin(2 * np.pi * df[col_name] / period)
    df[f'cos_{col_name}'] = np.cos(2 * np.pi * df[col_name] / period)
    df.drop(columns=[col_name], inplace=True)
    return df

def add_cyclical_features(df):
    for col in ['hour', 'dow', 'dom', 'doy', 'month', 'woy', 'year']:
        df = generate_cyclical_features(df, col)
    return df

def create_preprocessed_data(dhn_path, weather_path):
    df = load_and_merge_data(dhn_path, weather_path)
    df = add_time_features(df)
    df = add_holiday_weekend_features(df)
    df, scaler = scale_demand(df)
    dfs = df.copy()
    df_cycle = add_cyclical_features(df)
    return dfs, scaler, df_cycle

def plot_hourly_demand(df):
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='hour', y='Demand', hue='hour', ax=ax)
    ax.set_title('Consumption by Hour of Day')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Demand [kWh]')
    plt.show()

def plot_daily_trends(df):
    pivot_table = df.pivot_table(index='hour', columns='weekday', values='Demand', aggfunc='sum')
    fig, ax = plt.subplots()
    sns.lineplot(data=pivot_table, linewidth=2.5, ax=ax)
    ax.set_title('Daily Trends')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Demand [kWh]')
    ax.legend(frameon=False, prop={'size': 14})
    plt.show()

def plot_resampled_demand(df):
    data1 = df[['Demand']].copy()
    data1.index = pd.to_datetime(df.index)
    fig = plt.figure(figsize=(22, 20))
    fig.subplots_adjust(hspace=1)

    resample_periods = [('D', 'day'), ('W', 'week'), ('M', 'month'), ('Q', 'quarter'), ('A', 'year')]
    for i, (resample, label) in enumerate(resample_periods, start=1):
        ax = fig.add_subplot(5, 1, i)
        ax.plot(data1['Demand'].resample(resample).mean(), linewidth=1, color='purple')
        ax.set_title(f'Mean Energy consumption resampled over {label}')
        ax.tick_params(axis='both', which='major')
    plt.show()

def plot_quarterly_boxplots(df):
    for q in range(1, 5):
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.boxplot(x=df[df['quarter'] == q]['hour'], y=df[df['quarter'] == q]['Demand'], ax=ax)
        ax.set_title(f'Hourly Boxplot consumption in quarter {q}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Demand [kWh]')
        plt.show()

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    f, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        mask=mask,
        cmap=sns.diverging_palette(250, 10, as_cmap=True),
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0,
        cbar_kws={"shrink": .75},
        annot=False,
        ax=ax
    )
    ax.set_facecolor('white')
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(df):
    numeric_df = df.select_dtypes(include=[np.number])
    features = numeric_df.drop(columns=['Demand', 'Demand_scaled'], errors='ignore')
    target = df['Demand']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)
    importances = model.feature_importances_
    feature_names = features.columns
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.show()

def plot_pairplot(df):
    sns.pairplot(df, corner=True)
    plt.title("Pairwise Relationships")
    plt.show()


if __name__ == '__main__':
    df, scaler, df_cycle = create_preprocessed_data(file_path_demand, file_path_weather)
    df_pair = load_and_merge_data(file_path_demand,file_path_weather)
    plot_hourly_demand(df)
    plot_daily_trends(df)
    plot_resampled_demand(df)
    plot_quarterly_boxplots(df)
    plot_feature_importance(df_cycle)
    plot_correlation_matrix(df_cycle)
    plot_pairplot(df_pair)