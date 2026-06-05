import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
import kagglehub, shutil
import plotly.express as px
from yellowbrick.cluster import KElbowVisualizer #untuk visualisasi metode Elbow

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans, DBSCAN #untuk clustering
from sklearn.metrics import silhouette_score #untuk mengevaluasi hasil clustering
import os
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV

#fungsi preprocessing_lengkap
def preprocessing_pipeline(csv_path):
    df = pd.read_csv(csv_path):

    #Melihat negara mana saja yang memiki nilai nan pada gdp
    nan_gdp = df[df['gdp_usd'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on Gdp: ", nan_gdp)

    #Melihat negara mana saja yang memiki nilai nan pada wind_electricity
    nan_wind_twh = df[df['wind_electricity_twh'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on wind electricity: ", nan_wind_twh)

    #Melihat negara mana saja yang memiki nilai nan pada wind_electricity
    nan_wind_share = df[df['wind_share_pct'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on wind share: ", nan_wind_share)

    nan_hydro_twh = df[df['hydro_electricity_twh'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on hydro electricity: ", nan_hydro_twh)

    nan_solar_pct = df[df['solar_yoy_growth_pct'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on solar yoy growth: ", nan_solar_pct)

    nan_wind_pct = df[df['wind_yoy_growth_pct'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on wind yoy growth: ", nan_wind_pct)

    nan_renewable_pct = df[df['renewables_yoy_growth_pct'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on renewables yoy growth: ", nan_renewable_pct)

    nan_nuclear_twh = df[df['nuclear_electricity_twh'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on nuclear electricity: ", nan_nuclear_twh)

    nan_policy = df[df['policy_milestone'].isna()][['country','year']].drop_duplicates()
    print("Country with Nan on policy_milestone: ", nan_policy)

    #Mengatasi missing value
    df = df.sort_values(by=['country', 'year']).reset_index(drop=True)

    col_nan = ['gdp_usd', 'hydro_electricity_twh', 'solar_yoy_growth_pct', 'wind_yoy_growth_pct', 'renewables_yoy_growth_pct', 'nuclear_electricity_twh']

    for kolom in col_nan:
        df[kolom] = df.groupby('country')[kolom].ffill().bfill()

    df = df.replace([np.inf, -np.inf], 0)
    print(df.head())

    #Mengisi fillna dengan negara yang diketahui
    df = df.sort_values(by=['country', 'year']).reset_index(drop=True)
    spec_country = 'Singapore'
    nan_column = ['wind_electricity_twh', 'wind_share_pct']

    for kolom in nan_column:
        nilai_ffill = df.groupby('country')[kolom].ffill().bfill()

        df[kolom] = np.where((df['country'] == spec_country) & (df[kolom].isna()), 0, df[kolom])
        df[kolom] = df[kolom].fillna(nilai_ffill)

    df = df.replace([np.inf, -np.inf], 0)
    print(df.head())

    #Menghapus satu kolom
    df.drop(columns=['policy_milestone'], inplace=True)
    df.info()

    #Melihat Outiler
    num_fiture = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(20,20))
    for i, col in enumerate(num_fiture.columns, 1):
        plt.subplot(6, 5, i+1)
        sns.boxplot(y=df[col])
        plt.title(col)
    plt.tight_layout()
    plt.show()

    #Handling Outlier
    for col in num_fiture:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        #fungsi untuk mengganti outlier dengan batas atas dan batas bawah
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    df.head()

    #Binning diffence
    df['difference']=df['total_electricity_generation_twh'] - df['electricity_demand_twh']
    surp = df.difference[df.difference > 0]
    defi = df.difference[df.difference < 0]
    pas = df.difference[df.difference == 0]

    #menyusun data untuk plotting
    x = ["Surplus", "Defisit", "Pass"]
    y = [len(surp.values), len(defi.values), len(pas.values)]

    #membuat bar chart untuk distribusi selisih
    plt.figure(figsize=(15,6))
    plt.bar(x, y, color=['brown', 'green', 'blue'])
    plt.title("Selisih Penggunaan Energi Listrik dan Produksi Energi Listrik Setiap Negara")
    plt.xlabel(None)
    plt.ylabel("Summary")

    #Menambah label jumlah pelanggan diatas setiap bar
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha='center', va='bottom')

    plt.show()

    df['indicator'] =df.difference.apply(lambda x: "Surplus" if x > 0 else ("Defisit" if x < 0 else ("Pass")))
    df.head()

    #Normalisasi data
    #Menggunakan MinmaxSclaer karena outlier sudah ditangani
    scaler = MinMaxScaler()
    kol_scal = ['gdp_usd', 'total_electricity_generation_twh', 'electricity_demand_twh',
                'solar_electricity_twh', 'wind_electricity_twh', 'renewables_electricity_twh',
                'hydro_electricity_twh', 'nuclear_electricity_twh', 'fossil_electricity_twh',
                'solar_share_pct', 'wind_share_pct', 'renewables_share_pct', 'fossil_share_pct',
                'low_carbon_share_pct', 'carbon_intensity_gco2_kwh', 'co2_saved_solar_wind_mt',
                'solar_yoy_growth_pct', 'wind_yoy_growth_pct', 'renewables_yoy_growth_pct']

    for col in kol_scal:
        model = scaler.fit(df[[col]])
        df[col] = model.fit_transform(df[[col]])

    df.head()

    #Encoding
    categ_column =['income_group', 'indicator']

    le = LabelEncoder()

    for column in categ_column:
        df[column] = le.fit_transform(df[column])

    df.head()

    df_preprocessing=df

    return df_preprocessing

csv_path = 'https://raw.githubusercontent.com/ririhadis/workflow/main/global_renewable_energy_transition_2000_2025.csv'
df_preprocessing = preprocessing_pipeline(csv_path)
df_preprocessing.to_csv('data_preprocessing.csv', index=False)
    