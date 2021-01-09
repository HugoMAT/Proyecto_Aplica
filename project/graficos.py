import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import operator
import warnings
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")

preprocesado = pd.read_csv(os.path.join("data","Datos_Proyecto_Aplica.csv"))

def graphics1():
    preprocesado= pd.read_csv(os.path.join("data","Datos_Proyecto_Aplica.csv"))
    l=['runtime','revenue','budget','vote_average','vote_count','popularity']
    plt.figure(figsize=(18,8))
    j=0
    for i in l :
        j+=1
        plt.subplot(2,3,j)
        titulo="Histograma para la variable "+ i
        plt.title(titulo)
        plt.hist(preprocesado[i])
        
    plt.show

    
def graphics2():
    D=dict()
    D2=dict()
    for i in range(preprocesado.shape[0]):
        leng=ast.literal_eval(preprocesado.spoken_languages[i])
        for j in range(len(leng)):
            if leng[j] not in D.keys():
                D[leng[j]]=1
            else:  
                D[leng[j]]+=1
    leng_sort = sorted(D.items(), key=operator.itemgetter(1), reverse=True)
    for name in enumerate(leng_sort):
        D2[name[1][0]] = D[name[1][0]]              
    df = pd.DataFrame([[key, D2[key]] for key in D2.keys()], columns=['Lenguaje', 'Frecuencia']).iloc[:10,:]    
    df.iloc[6][0]='Chino Mandarin'
    df.iloc[7][0]='Japones'
    plt.barh(df['Lenguaje'],df['Frecuencia'])

def graphics3():
    pop= preprocesado.sort_values('popularity', ascending=False)
    plt.figure(figsize=(12,4))
    plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
            color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("Popularidad")
    plt.title("Peliculas más populares")
    plt.show()
    
def kme():
    
    numeric_features = list(preprocesado.dtypes[preprocesado.dtypes != "object"].index)
    numericas_prepro=preprocesado[numeric_features].iloc[:,1:].drop(["id","vote_average","vote_count"],axis=1)

    #imputamos por el promedio los dos perdidos
    is_NaN = numericas_prepro.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = numericas_prepro[row_has_NaN]
    numericas_prepro.loc[numericas_prepro[row_has_NaN].index,"runtime"]= numericas_prepro.runtime.mean()
    
    Nc = range(1, 10)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    score = [kmeans[i].fit(numericas_prepro.iloc[:,1:]).inertia_ for i in range(len(kmeans))]


    df_Elbow = pd.DataFrame({'Number of Clusters':Nc,
                            'Score':score})

    df_Elbow.head()
    # graficar los datos etiquetados con k-means
    fig, ax = plt.subplots(figsize=(11, 8.5))
    plt.title('Elbow Curve')
    sns.lineplot(x="Number of Clusters",
                 y="Score",
                data=df_Elbow)
    sns.scatterplot(x="Number of Clusters",
                 y="Score",
                 data=df_Elbow)