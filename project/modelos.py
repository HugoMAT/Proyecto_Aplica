import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import lectura          as le
import preprocesamiento as pr
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

credits, movies = le.lecture()
preprocesado = pr.preprocesamiento(credits, movies)


def lista_recomendados(idx , col):
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    preprocesado[col] = preprocesado[col].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(preprocesado[col])
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return sim_scores


### Para Kmeans

numeric_features = list(preprocesado.dtypes[preprocesado.dtypes != "object"].index)
numericas_prepro=preprocesado[numeric_features].iloc[:,1:].drop(["id","vote_average","vote_count"],axis=1)

#imputamos por el promedio los dos perdidos
is_NaN = numericas_prepro.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = numericas_prepro[row_has_NaN]
numericas_prepro.loc[numericas_prepro[row_has_NaN].index,"runtime"]= numericas_prepro.runtime.mean()

# implementación de la regla del codo
Nc = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
score = [kmeans[i].fit(numericas_prepro.iloc[:,1:]).inertia_ for i in range(len(kmeans))]


df_Elbow = pd.DataFrame({'Number of Clusters':Nc,
                        'Score':score})
###modelo kmeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(numericas_prepro.iloc[:,1:])



def top_inter(indice_pelicula):
    top=[]
    generos_pelicula    = ast.literal_eval(preprocesado.genres[indice_pelicula]              )
    prod_pelicula       = ast.literal_eval(preprocesado.production_companies[indice_pelicula])
    prod_pais_pelicula  = ast.literal_eval(preprocesado.production_countries[indice_pelicula])
    leng_pelicula       = ast.literal_eval(preprocesado.spoken_languages[indice_pelicula]    )
    
    for i in range(preprocesado.shape[0]):
        generos_2_peliculas   = ast.literal_eval(preprocesado.genres[i]              )
        prod_2_peliculas      = ast.literal_eval(preprocesado.production_companies[i])
        prod_pais_2_peliculas = ast.literal_eval(preprocesado.production_countries[i])
        leng_2_pelicula       = ast.literal_eval(preprocesado.spoken_languages[i]    )
        
        
        interseccion_gen       = len(list(set(generos_pelicula)   & set(generos_2_peliculas  )))
        interseccion_prod      = len(list(set(prod_pelicula)      & set(prod_2_peliculas     )))
        interseccion_prod_pais = len(list(set(prod_pais_pelicula) & set(prod_pais_2_peliculas)))
        interseccion_leng      = len(list(set(leng_pelicula)      & set(leng_2_pelicula      )))
        
        top.append([i,preprocesado.title[i],interseccion_gen, int(interseccion_prod),int(interseccion_prod_pais), int(interseccion_leng)])
        
    top_df =pd.DataFrame(data=np.array(top),columns=["pelicula_fila","title","generos_en_comun","productora_en_comun","pais_prod_en_comun","idiomas_en_comun"])
    top_df["productora_en_comun"]=top_df["productora_en_comun"].astype(int)
    top_df["generos_en_comun"]=top_df["generos_en_comun"].astype(int)
    top_df["pelicula_fila"]=top_df["pelicula_fila"].astype(int)
    
    return top_df[(top_df["generos_en_comun"] >0 )& (top_df["productora_en_comun"]> 0 )]


def fechas_limites(indice_pelicula):
    peliculas_cercanas=[]
    fecha=pd.to_datetime(preprocesado.release_date)
    fecha_pelicula= fecha[indice_pelicula]
    for i in range(fecha.shape[0]):
        if fecha[i].year +15 >fecha_pelicula.year and fecha[i].year-15 <fecha_pelicula.year:
            peliculas_cercanas.append([i,fecha[i],preprocesado.title[i]])
    peliculas_cercanas =pd.DataFrame(data=np.array(peliculas_cercanas),columns=["pelicula_fila","fecha","title"])
    return peliculas_cercanas
            


def find_director(indice_pelicula):
    crew_pel = ast.literal_eval(preprocesado.job[indice_pelicula])
    for i in range(len(crew_pel)):
        if crew_pel[i] == 'Director':
            break
    director_por_buscar = ast.literal_eval(preprocesado.name_crew[indice_pelicula])[i]
    df_aux= preprocesado[preprocesado[['name_crew']].applymap(lambda x: director_por_buscar in x ).name_crew]
    return df_aux[['title', 'Unnamed: 0']].rename(columns = {'Unnamed: 0': 'pelicula_fila'})
    
def find_actores(indice_pelicula):
    aux = []
    actores = set(ast.literal_eval(preprocesado.name_cast[indice_pelicula])[:10])
    for i in range(preprocesado.shape[0]):
        condicion = list((i,preprocesado.title[i], len(actores & set(ast.literal_eval(preprocesado.name_cast[i])[:10]))))
        if preprocesado.title[i] != preprocesado.title[indice_pelicula]:
            aux.append( condicion )
    df_aux = pd.DataFrame(data = np.array(aux), columns= ['pelicula_fila','title','actores_comunes'])
    df_aux["actores_comunes"] = df_aux["actores_comunes"].astype(int)
    df_aux["pelicula_fila"] = df_aux["pelicula_fila"].astype(int)
    return  df_aux[ df_aux.actores_comunes > 0]



def juntar(df1, df2, df3):
    #Asumimos que df1: top_inter, df2: find_actores, df3: find_director
    pel_fil = set()
    data= pd.DataFrame(columns=['pelicula_fila', 'title', 'generos_en_comun', 'productora_en_comun',
       'pais_prod_en_comun', 'idiomas_en_comun', 'actores_comunes', 'director_en_comun'])
    
    for x in df1.pelicula_fila:
        pel_fil.add(x)
    
    for x in df2.pelicula_fila:
        pel_fil.add(x)
        
    for x in df3.pelicula_fila:
        pel_fil.add(x)
    
    for i in list(pel_fil):
        if i in list(df1.pelicula_fila):
            a= list(df1.loc[i, ['pelicula_fila', 'title', 'generos_en_comun', 'productora_en_comun',
       'pais_prod_en_comun', 'idiomas_en_comun']])
            if i in list(df2.pelicula_fila):
                a= a+ list(df2[df2['pelicula_fila']==i].actores_comunes)
                if i in list(df3.pelicula_fila):
                    a+=[1]
                else:
                    a+=[0]
            else:
                a+= [0]
                if i in list(df3.pelicula_fila):
                    a+=[1]
                    data.loc[i]=a
                else:
                    a+=[0]
                    data.loc[i]=a
        else:
            if i in list(df2.pelicula_fila):
                a= [i] + list(df2[df2['pelicula_fila']==i].title) +[0, 0, 0 ,0]
            else:
                a= [i] + list(df3[df3['pelicula_fila']==i].title) +[0, 0, 0 ,0]
            if i in list(df2.pelicula_fila):
                a+= list(df2[df2['pelicula_fila']==i].actores_comunes)
                if i in list(df3.pelicula_fila):
                    a+=[1]
                    data.loc[i]=a
                else:
                    a+=[0]
                    data.loc[i]=a
            else:
                a+= [0]
                if i in list(df3.pelicula_fila):
                    a+=[1]
                    data.loc[i]=a
                else:
                    a+=[0]
                    data.loc[i]=a
     
    return data



def concatenar(indice_pelicula):
    df_aux = juntar(top_inter(indice_pelicula), find_actores(indice_pelicula), find_director(indice_pelicula))
    for i in df_aux.columns[2:-2]:
        df_aux[i] = df_aux[i].astype(int)
    
    df_aux['score'] = df_aux[['generos_en_comun', 'productora_en_comun', 'pais_prod_en_comun', 'idiomas_en_comun','actores_comunes']].sum(axis = 1)
    
    return df_aux[['pelicula_fila','title','score', 'director_en_comun']].sort_values(by=['score'], ascending=False)



def text_score(idx):
    result=list()
    L_o=lista_recomendados(idx,'overview')
    mov_over = set([i[0] for i in L_o])
    score_over= [i[1] for i in L_o]
    L_k=lista_recomendados(idx,'keywords')
    mov_key = set([i[0] for i in L_k])
    score_key=[i[1] for i in L_k]
    L_mt=lista_recomendados(idx,'movies+tagline')
    mov_mt = set([i[0] for i in L_mt])
    score_mt=[i[1] for i in L_mt]
    mov_fin = mov_over|mov_key|mov_mt
    if idx in list(mov_fin):
        mov_fin.remove(idx)
    for i  in mov_fin:
        text=0
        if i in mov_over:
            if 1/score_over[list(mov_over).index(i)] == float('inf'):
                text=20
            else:
                text+=1/score_over[list(mov_over).index(i)]
        if i in mov_key:
            if 1/score_key[list(mov_key).index(i)] == float('inf'):
                text=20
            else:
                text+=1/score_over[list(mov_key).index(i)]
        if i in mov_mt:
            if 1/score_mt[list(mov_mt).index(i)] == float('inf'):
                text=20
            else:
                text+=1/score_over[list(mov_mt).index(i)]
        result.append([i,text/3])
    
    data= [[i[1]] for i in result]
    scaler = StandardScaler()
    scaler.fit(data)
    escalado = scaler.transform(data)
    
    escalado_2 = [i[0] for i in escalado]
    
    df = pd.DataFrame()
    
    df['pelicula_fila'] = list(mov_fin)
    df['score'] = escalado_2
    return df


def Comunes(indice_pelicula):
    scaler = StandardScaler()
    df_aux = concatenar(indice_pelicula)
    mask = list(df_aux.director_en_comun > 0)
    df_aux.loc[mask,('score')] = df_aux.score[mask] * 1.05
    data = [[i] for i in df_aux.score]
    scaler.fit(data)
    df_aux.score = scaler.transform(data)
    return df_aux

def KMeans_s(indice_pelicula):
  
    mask = []
    cluster_pelicula = kmeans.predict([numericas_prepro.iloc[indice_pelicula,1:]])[0]
    for i in range(numericas_prepro.shape[0]):
        if kmeans.predict([numericas_prepro.iloc[i,1:]])[0] == kmeans.predict([numericas_prepro.iloc[indice_pelicula,1:]])[0]:
            mask.append(True)
        else:
            mask.append(False)
    return numericas_prepro.iloc[mask].index


def recomendador_final(nombre_pelicula):
    indice_pelicula = preprocesado.title.values.tolist().index(nombre_pelicula)
    dic = dict()
    t_s = text_score(indice_pelicula)
    com = Comunes(indice_pelicula)
    km= KMeans_s(indice_pelicula)
    fl = fechas_limites(indice_pelicula)
    lenguaje= preprocesado.original_language[indice_pelicula]
    
    for i in range(preprocesado.shape[0]):
        
        if i in list(t_s.pelicula_fila):
            dic[i] = t_s[t_s['pelicula_fila']==i].score.values[0]*0.7
            
        if i in com.pelicula_fila:
            if i in dic.keys():
                dic[i] += com[com['pelicula_fila']==i].score.values[0]*0.6
            else:
                dic[i] = com[com['pelicula_fila']==i].score.values[0]*0.6
        
        if i in list(km) :
            if i in dic.keys():
                dic[i] += dic[i]*0.3

        if i in list(fl.pelicula_fila):
            if i in dic.keys():
                dic[i] += dic[i]*0.1
            
        if lenguaje == preprocesado.original_language[i]:
            if i in dic.keys():
                dic[i] += dic[i]*0.05
    df = pd.DataFrame()
    
    df['pelicula_fila'] = list(dic.keys())
    df['scores'] = list(dic.values())
    df_f= df.sort_values(by=['scores'], ascending=False).iloc[:20,:]
    df_f['title'] = preprocesado.title[df_f.pelicula_fila].values
    
    indice = df_f[df_f['title']==nombre_pelicula].index.item()
    df_f = df_f.drop([indice],axis=0)
    
    
    return df_f