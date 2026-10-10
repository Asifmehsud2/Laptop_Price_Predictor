import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import xgboost as xgb


df = pd.read_csv('laptop_data.csv')
df = df.drop('Unnamed: 0',axis=1)
df['Ram'] = df['Ram'].str.replace("GB","").astype(int)
df['Weight'] = df['Weight'].str.replace("kg","").astype(float)
df["IPs"] = df['ScreenResolution'].str.contains("IPS").astype(int)
df["TouchScreen"] = df['ScreenResolution'].str.contains("Touch").astype(int)
df["X_res"] = df["ScreenResolution"].str.extract(r'(\d+)x').astype(int)

df["Y_res"] = df["ScreenResolution"].str.extract(r'x(\d+)').astype(int)

df.drop(columns=("ScreenResolution"),inplace=True)
df["CPU_Brand"] = df["Cpu"].apply(lambda x: x.split()[0])

df["ClockSpeed"] = df["Cpu"].str.extract(r'(\d+\.\d+)GHz').astype(float)

df.drop(columns="Cpu", inplace=True)
df["GPU_Brand"] = df["Gpu"].apply(lambda x: x.split()[0])
df.drop(columns="Gpu", inplace=True)

categorical_cols = [
    "Company",
    "TypeName",
    "OpSys",
    "CPU_Brand",
    "GPU_Brand"
]

df = pd.get_dummies(
    df,
    columns=categorical_cols,
    drop_first=True
)

df[df.select_dtypes("bool").columns] = (
    df.select_dtypes("bool").astype(int)
)

df["Memory"] = df["Memory"].str.replace("TB", "000GB")

df["SSD"] = 0
df["HDD"] = 0
df["Flash"] = 0
df["Hybrid"] = 0


import re

for i, memory in enumerate(df["Memory"]):

    if "SSD" in memory:
        match = re.search(r"(\d+)GB SSD", memory)
        if match:
            df.loc[i, "SSD"] = int(match.group(1))

    if "HDD" in memory:
        match = re.search(r"(\d+)GB HDD", memory)
        if match:
            df.loc[i, "HDD"] = int(match.group(1))

    if "Flash Storage" in memory:
        match = re.search(r"(\d+)GB Flash Storage", memory)
        if match:
            df.loc[i, "Flash"] = int(match.group(1))

    if "Hybrid" in memory:
        match = re.search(r"(\d+)GB Hybrid", memory)
        if match:
            df.loc[i, "Hybrid"] = int(match.group(1))

df.drop(columns="Memory", inplace=True)
df["Total_Storage"] = (
    df["SSD"] +
    df["HDD"] +
    df["Flash"] +
    df["Hybrid"]
)

x = df.drop('Price',axis= 1)
y =df['Price']
x['ClockSpeed'] = x['ClockSpeed'].fillna(df['ClockSpeed'].mean())
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2 ,random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = xgb.XGBRegressor(
    n_estimators=400,          
    max_depth=6,               
    learning_rate=0.05,        
    subsample=0.8,             
    colsample_bytree=0.8,      
    min_child_weight=2,    
    reg_alpha=0.1,            
    reg_lambda=1.0,            
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

import joblib

joblib.dump(model, "model.joblib")

print("R2 Score:", r2_score(y_test, y_pred))