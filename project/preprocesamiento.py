# Lectura de Datos:

# Importamos las librerías apropiadas

import numpy     as np
import pandas    as pd
import missingno as msno
from lectura import lecture
from ast import literal_eval

def preprocesamiento(credits, movies):
    """
        preprocesamiento()

        Trabaja transformando las columnas cast, crew del DataFrame credits, separando cada
        llave y valor de los diccionarios en nuevas columnas respetando el orden; y las columnas
        genres, keywords, production_companies, production_countries y spoken_languages del
        DataFrame movies, separando cada llave y valor en columnas nuevas, para así luego
        combinarlas en un sólo DataFrame, eliminando la columna title que se encuentra repetida,
        homepage, status, original_title y además, juntamos la columna de title, con tagline, para
        crear una sola columna llamada movies+tagline.

        Parameters
        ----------
            credits : pandas.DataFrame
            movies  : pandas.DataFrame 
      
        Returns
        -------
        movie_credits : pandas.DataFrame
            DataFrame procesado combinación entre credits y movies.

    """    
    print("Empezando procesamiento de credits.")
    # Eliminamos dicha columna pues se causará error más adelante pues coincide con el nombre que tendrá una columna de credits
    movies  = movies.drop('title',axis=1)

    # Lectura y separación de datos en _Credits_.
    # Aplicaremos Literal_eval( ) a cada registro de la columna **cast** y **crew** respectivamente, para luego mantener la agrupación por **movie_id** y generar un único DataFrame con todos los datos.

    # Para Cast
    data_frame_1 = []
    data_frame_2 = []
    data_frame_3 = []
    data_frame_4 = []
    data_frame_5 = []
    data_frame_6 = []
    data_frame_7 = []

    for i in range(len(credits['cast'])):
        cast_id   = []
        character = []
        credit_id = []
        gender    = []
        idd       = []
        name      = []
        order     = []
        for dicci in literal_eval(credits['cast'][i]):
            cast_id_t , character_t, credit_id_t, gender_t, idd_t, name_t, order_t = dicci.items()

            cast_id.append  ( cast_id_t[1]   )
            character.append( character_t[1] )
            credit_id.append( credit_id_t[1] )
            gender.append   ( gender_t[1]    )
            idd.append      ( idd_t[1]       )
            name.append     ( name_t[1]      )
            order.append    ( order_t[1]     )

        data_frame_1.append(cast_id  )
        data_frame_2.append(character)
        data_frame_3.append(credit_id)
        data_frame_4.append(gender   )
        data_frame_5.append(idd      )
        data_frame_6.append(name     )
        data_frame_7.append(order    )

    credits = credits.assign(cast_id = data_frame_1, character = data_frame_2, credit_id_cast = data_frame_3, gender_cast = data_frame_4, id_cast = data_frame_5, name_cast = data_frame_6, order = data_frame_7)

    # Para Crew
    data_frame_1 = []
    data_frame_2 = []
    data_frame_3 = []
    data_frame_4 = []
    data_frame_5 = []
    data_frame_6 = []
    for i in range(len(credits['crew'])):
        credit_id  = []
        department = []
        gender     = []
        idd        = []
        job        = []
        name       = []
        for dicci in literal_eval(credits['crew'][i]):
            credit_id_t , department_t, gender_t, idd_t, job_t, name_t = dicci.items()

            credit_id.append ( credit_id_t[1]  )
            department.append( department_t[1] )
            gender.append    ( gender_t[1]     )
            idd.append       ( idd_t[1]        )
            job.append       ( job_t[1]        )
            name.append      ( name_t[1]       )

        data_frame_1.append(credit_id  )
        data_frame_2.append(department )
        data_frame_3.append(gender     )
        data_frame_4.append(idd        )
        data_frame_5.append(job        )
        data_frame_6.append(name       )

    credits = credits.assign(credit_id_crew = data_frame_1, department = data_frame_2, gender_crew = data_frame_3, id_crew = data_frame_4, job = data_frame_5, name_crew = data_frame_6).drop(["cast","crew"], axis=1)
    print("Procesamiento de Credits listo!")
    # Lectura y separación de datos en _Movies_.
    # Aplicaremos _Literal_eval( )_ a cada registro de la columna **genres**, **keywords**, **production_companies**, **production_countries** y **spoken_languages** respectivamente, para luego mantener la agrupación por **movie_id** y generar un único DataFrame con todos los datos.
    print("Empezando procesamiento de movies.")

    # Aplicación a Géneros.
    data_frame_1 = []
    data_frame_2 = []

    for i in range(len(movies['genres'])):
        ids = []
        kw  = []
        for dicci in literal_eval(movies['genres'][i]):
            i_d , k_w = dicci.items()
            ids.append( i_d[1] )
            kw.append ( k_w[1] )

        data_frame_1.append(ids)
        data_frame_2.append(kw)

    movies = movies.assign(id_genres = data_frame_1, genres = data_frame_2)


    # Aplicación a Keywords.

    data_frame_1 = []
    data_frame_2 = []

    for i in range(len(movies['keywords'])):
        ids = []
        kw  = []
        for dicci in literal_eval(movies['keywords'][i]):
            i_d , k_w = dicci.items()
            ids.append( i_d[1] )
            kw.append ( k_w[1] )

        data_frame_1.append(ids)
        data_frame_2.append(kw)

    movies = movies.assign(id_key = data_frame_1, keywords = data_frame_2)


    # Aplicación a Compañías Productoras.

    data_frame_1 = []
    data_frame_2 = []

    for i in range(len(movies['production_companies'])):
        ids = []
        kw  = []
        for dicci in literal_eval(movies['production_companies'][i]):
            i_d , k_w = dicci.items()
            ids.append( i_d[1] )
            kw.append ( k_w[1] )
        data_frame_1.append(ids)
        data_frame_2.append(kw)

    movies = movies.assign(production_companies = data_frame_1, id_production_companies = data_frame_2)

    # Aplicación a Países donde fue producida.

    data_frame_1 = []
    data_frame_2 = []

    for i in range(len(movies['production_countries'])):
        ids = []
        kw  = []

        for dicci in literal_eval(movies['production_countries'][i]):
            i_d , k_w = dicci.items()
            ids.append( i_d[1] )
            kw.append ( k_w[1] )

        data_frame_1.append(ids)
        data_frame_2.append(kw)

    movies = movies.assign(iso_production_countries = data_frame_1, production_countries = data_frame_2)

    # Aplicación a Lenguaje hablados en la película.


    data_frame_1 = []
    data_frame_2 = []

    for i in range(len(movies['spoken_languages'])):
        ids = []
        kw  = []
        for dicci in literal_eval(movies['spoken_languages'][i]):
            i_d , k_w = dicci.items()
            ids.append( i_d[1] )
            kw.append ( k_w[1] )
        data_frame_1.append(ids)
        data_frame_2.append(kw)

    movies = movies.assign(iso_spoken_languages = data_frame_1, spoken_languages = data_frame_2)
    print("Procesamiento de Movies listo!")

    # Unificamos todo en un sólo DataFrame.

    movie_credits = pd.concat((credits, movies), axis=1)

    # Fabricamos una nueva columna llamada **movies+tagline**, el cual reune el **título** de la película, además del **tagline**, borrando **tagline** que ya estará contemplada en la que creamos y **homepage** y **status** que son completamentente innecesarias a nuestro criterio.


    movie_credits["movies+tagline"] = movie_credits.loc[movie_credits['tagline'].notna(), "title" ] +', ' + movie_credits.loc[movies['tagline'].notna(), "tagline" ]

    movie_credits.loc[movie_credits.runtime == 0, ["runtime"] ] = movie_credits.runtime.median()
    movie_credits = movie_credits.drop(['tagline','homepage','original_title','status'], axis=1)

    # Por último, guardamos en memoria el DataFrame limpio.
    movie_credits.to_csv("Datos_Proyecto_Aplica.csv")
    
    return movie_credits
