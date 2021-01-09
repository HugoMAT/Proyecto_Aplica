# Lectura de Datos:
import os
import pandas as pd

def lecture():
    """
        lecture()

            Lee archivos "tmdb_5000_credits.csv" y "tmdb_5000_movies.csv" 
            para luego transformarlos en pandas.DataFrame.

        Parameters
        ----------
            None.

        Returns
        -------
            credits, movies : [pandas.DataFrame, pandas.DataFrame]
                Es una lista del DataFrame credits y DataFrame movies.

    """
    try:
        credits = pd.read_csv(os.path.join("data","tmdb_5000_credits.csv"))
        movies  = pd.read_csv(os.path.join("data","tmdb_5000_movies.csv"))
        print("Datos correctamente leídos")
        return credits, movies
    except:
        print("Datos incorrectamente leídos")
        return None, None