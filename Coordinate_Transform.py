from pyproj import Proj
import pandas as pd


df=pd.read_excel("List of Anomalies_Rosen.xlsx")
df.head()

myProj=Proj('+proj=utm +zone=23 +south +ellps=WGS84',preserve_units=False)
df['long'],df['lat']=myProj(df['coordenadas_x'].values,df['coordenadas_y'].values, inverse=True)
df.to_excel('Coordenadafinal.xlsx')

print(df.head(10))