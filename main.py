import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

app = FastAPI()

# Importaion de datos
df_developer = pd.read_parquet("data/developer.parquet")
df_userdata = pd.read_parquet('data/userdata.parquet')
df_best_developer = pd.read_parquet('data/UsersbestDeveloper.parquet')
df_PlayTimeGenre = pd.read_parquet("data/play_time_genres.parquet")
df_UserForGenre_parte_1 = pd.read_parquet("data/user_for_genre_part_1.parquet")
df_UserForGenre_parte_2 = pd.read_parquet("data/user_for_genre_part_2.parquet")
df_UsersRecommend = pd.read_parquet("data/UsersRecommend.parquet")
df_UsersWorstDeveloper = pd.read_parquet("data/UsersWorstDeveloper.parquet")
df_Sentiment_Analysis = pd.read_parquet("data/sentiment_analysis.parquet")


@app.get(path="/", response_class=HTMLResponse,tags=["Home"])
def presentacion():
    '''
    Genera una página de presentación HTML para la API Steam de consultas de videojuegos.
    
    Returns:
    str: Código HTML que muestra la página de presentación.
    '''
    return '''
    <html>
        <head>
            <title>API Steam</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }
                h1 {
                    color: #333;
                    text-align: center;
                }
                p {
                    color: #666;
                    text-align: center;
                    font-size: 18px;
                    margin-top: 20px;
                }
            </style>
        </head>
        <body>
            <h1>API de consultas de videojuegos de la plataforma Steam</h1>
            <p>Bienvenido a la API de Steam donde se pueden hacer diferentes consultas sobre la plataforma de videojuegos.</p>
            <p>INSTRUCCIONES:</p>
            <p>Escriba <span style="background-color: lightgray;">/docs</span> a continuación de la URL actual de esta página para interactuar con la API</p>
            <p> Visita mi perfil en <a href="https://www.linkedin.com/in/frank-urribarri-0a4b47290/"><img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-blue?style=flat-square&logo=linkedin"></a></p>
            <p> El desarrollo de este proyecto esta en <a href="https://github.com/FrankJZx/PI_ML_OPS_Steam"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-black?style=flat-square&logo=github"></a></p>
        </body>
    </html>
    '''

### correccion, funciones 0.1

@app.get("/Developer")
def Developer(nombre_desarrollador):
    """
    ingresa numbre de desarrollador
    devuelve: Cantidad de items y porcentaje de contenido Free según empresa desarrolladora
    """
    try:
        # Filtrar el DataFrame por el nombre del desarrollador
        desarrollador_filtrado = df_developer[df_developer["developer"] == str(nombre_desarrollador)]
        
        if desarrollador_filtrado.empty:
            return {"error": "No se encontró ningún desarrollador con ese nombre."}
        
        # Obtener la cantidad de juegos del desarrollador filtrado
        cantidad_juegos = desarrollador_filtrado["Cantidad de Juegos por Developer"].iloc[0]
        
        # Obtener el porcentaje de juegos gratuitos del desarrollador filtrado
        porcentaje_free = desarrollador_filtrado["Porcentaje de Juegos Gratuitos"].iloc[0]
        
        resultado = {
                "Nombre del desarrollador": nombre_desarrollador,
                "Cantidad de juegos desarrollados: " : int(cantidad_juegos),
                "Porcentaje de ellos free: ": round(float(porcentaje_free),2)
                    }

        return resultado
    
    except Exception as e:

        return {"error": str(e)}


@app.get("/Userdata")
def userdata(user_id : object):
    """
    Devuelve cantidad de dinero gastado por el usuario, el porcentaje de recomendación y cantidad de items.
    """
    # Filtrar el DataFrame por user_id
    usuario_filtrado = df_userdata[df_userdata['user_id'] == str(user_id)]

    # Obtener la cantidad de dinero gastado por el usuario
    dinero_gastado = usuario_filtrado['total_price_por_usuario'].iloc[0]
    
    # Obtener el porcentaje de recomendación en base a reviews.recommend
    porcentaje_recomendacion = usuario_filtrado['porcentaje_recomendacion'].iloc[0]
    
    # Obtener la cantidad de items
    cantidad_items = usuario_filtrado['cantidad_total_juegos_por_usuario'].iloc[0]
    
    resultados = {
        "El user_id: ": user_id,
        "gasto en total: ": int(dinero_gastado),
        "porcentaje de recomendacion": round(float(porcentaje_recomendacion),2),
        "y compro en total:": int(cantidad_items)
    }
    
    return resultados


# Primera funcion: PlaytimeGenre

@app.get("/PlayTimeGenre")
def PlayTimeGenre( genero : str ):
    """
    Funcion que devuelve el año con mas horas jugadas para dicho género.
    """
    generos = df_PlayTimeGenre[df_PlayTimeGenre["genres"] == genero] 
    if generos.empty: 
        return f"No se encontraron datos para el género {genero}"
    año_max = generos.loc[generos["playtime_forever"].idxmax()]
    result = {
        'Genero': genero,
        'Año con Más Horas Jugadas': int(año_max["year"]),
        'Total de Horas Jugadas': año_max["playtime_forever"]
    }

    return result

# Segunda funcion: UserForGenre parte 1

@app.get("/UserForGenre_parte1")
def UserForGenre(genero: str):
    """
    Función que devuelve el usuario que acumula más horas jugadas para el género dado 
    y el total de horas jugadas para ese género.
    """
    # Filtrar el DataFrame por el género dado
    generos2 = df_UserForGenre_parte_1[df_UserForGenre_parte_1["genres"] == genero]
    
    # Obtener el usuario con más horas jugadas para ese género
    user_max = generos2.loc[generos2["playtime_forever"].idxmax()]["user_id"]
    
    # Calcular el total de horas jugadas para ese género
    horas_total = generos2["playtime_forever"].sum()

    # Crear el diccionario de resultados
    result = {
        "Genero": genero,
        "Usuario con Más Horas Jugadas": user_max,
        "Total de Horas Jugadas": horas_total
    }
    
    return result

# Segunda funcion: UserForGenre parte 2

@app.get("/UserForGenre_parte2")
def UserForGenre2(genero: str):
    """
    Función que devuelve un DataFrame filtrado por el género dado y las horas jugadas por año para ese género.
    """
    # Filtrar el DataFrame por el género dado
    generos3 = df_UserForGenre_parte_2[df_UserForGenre_parte_2["genres"] == genero]
   
    # Calcular las horas jugadas por año para ese género
    horas_x_año3 = generos3["playtime_forever"].sum()
    
    # Crear el DataFrame con los datos filtrados y las horas jugadas por año
    df_result = pd.DataFrame({
        "Genero": [genero],
        "Total de Horas Jugadas hasta ahora": [horas_x_año3]
    })

    return df_result

@app.get("/Best_developer")
def best_developer_year( año : int ):
    """
    Funcion que devuelve el top 3 de desarrolladoras con juegos MENOS 
    recomendados por usuarios para el año dado.
    """
    df_año2 = df_best_developer[df_best_developer["anio"]== año]
    if type(año) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if año < df_best_developer["anio"].min() or año > df_best_developer["anio"].max():
        return {"Año no encontrado "}
    df_ordenado_recomendacion2 = df_año2.sort_values(by="num_reviews_positivas", ascending=False)
    top_3_developers = df_ordenado_recomendacion2[["developer","num_reviews_positivas"]]
    result4 = {
        'Año': año,
        'Top 3 Desarrolladoras mas Recomendados': top_3_developers.to_dict(orient="records")
    }
    return result4



# Tercera funcion: UsersRecommend

@app.get("/UsersRecommend")
def UsersRecommend( año : int ):
    """
    Funcion que devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado.
    """
    df_año= df_UsersRecommend[df_UsersRecommend["anio"]== año]
    if type(año) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if año < df_UsersRecommend["anio"].min() or año > df_UsersRecommend["anio"].max():
        return {"Año no encontrado"}
    df_ordenado_recomendacion = df_año.sort_values(by="num_reviews_positivas", ascending=False)
    top_3_juegos = df_ordenado_recomendacion.head(3)[["app_name","num_reviews_positivas"]]
    result3 ={
        "Año": año,
        "Top 3 Juegos Más Recomendados": top_3_juegos.to_dict(orient='records')
    }
    return result3

# Cuarta funcion: UsersWorstDeveloper

@app.get("/UsersWorstDeveloper")
def UsersWorstDeveloper( año : int ):
    """
    Funcion que devuelve el top 3 de desarrolladoras con juegos MENOS 
    recomendados por usuarios para el año dado.
    """
    df_año2 = df_UsersWorstDeveloper[df_UsersWorstDeveloper["anio"]== año]
    if type(año) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if año < df_UsersRecommend["anio"].min() or año > df_UsersRecommend["anio"].max():
        return {"Año no encontrado "}
    df_ordenado_recomendacion2 = df_año2.sort_values(by="num_reviews_negativas", ascending=False)
    top_3_developers = df_ordenado_recomendacion2[["developer","num_reviews_negativas"]]
    result4 = {
        'Año': año,
        'Top 3 Desarrolladoras Menos Recomendadas': top_3_developers.to_dict(orient="records")
    }
    return result4


# Quinta funcion : sentiment_analysis

@app.get("/SentimentAnalysis")
def sentiment_analysis( desarrollador : str ):
    """
    Funcion que devuelve un diccionario con el nombre de la desarrolladora como llave y una lista 
    con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con 
    un análisis de sentimiento como valor.
    """
    if type(desarrollador) != str:
        return "Debes colocar un developer de tipo str, EJ:'07th Expansion'"
    if len(desarrollador) == 0:
        return "Debes colocar un developer en tipo String"
    
    # Filtrar el DataFrame por el desarrollador dado
    df_desarrollador = df_Sentiment_Analysis[df_Sentiment_Analysis["developer"] == desarrollador]
    
    # Sumar el número total de reseñas negativas, neutrales y positivas
    reseñas_negativas = df_desarrollador["cant_sentiment_negative"].sum()
    reseñas_neutrales = df_desarrollador["cant_sentiment_neutral"].sum()
    reseñas_positivas = df_desarrollador["cant_sentiment_positive"].sum()

    # Crear el diccionario de resultados
    resultados = {
        "El desarrollador: ": desarrollador,
        "tiene de reseñas Negativas": reseñas_negativas,
        "tiene de reseñas Neutrales": reseñas_neutrales,
        "tiene de reseñas Positivas": reseñas_positivas
    }
    
    return resultados

# Sexta funcion: Sistema de recomendacion de juegos

modelo_railway = pd.read_parquet("data/modelo_railway.parquet")

@app.get("/recomendacion_juego/{id}", name= "RECOMENDACION_JUEGO")
async def recomendacion_juego(id: int):
    
    """La siguiente funcion genera una lista de 5 juegos similares a un juego dado (id)
    Parametros:
    El id del juego para el que se desean encontrar juegos similares. Ej: 10
    Retorna:
    Un diccionario con 5 juegos similares 
    """
    game = modelo_railway[modelo_railway['id'] == id]

    if game.empty:
        return("El juego '{id}' no posee registros.")
    
    # Obtiene el índice del juego dado
    idx = game.index[0]

    # Toma una muestra aleatoria del DataFrame df_games
    sample_size = 2000  # Define el tamaño de la muestra (ajusta según sea necesario)
    df_sample = modelo_railway.sample(n=sample_size, random_state=42)  # Ajusta la semilla aleatoria según sea necesario

    # Calcula la similitud de contenido solo para el juego dado y la muestra
    sim_scores = cosine_similarity([modelo_railway.iloc[idx, 3:]], df_sample.iloc[:, 3:])

    # Obtiene las puntuaciones de similitud del juego dado con otros juegos
    sim_scores = sim_scores[0]

    # Ordena los juegos por similitud en orden descendente
    similar_games = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != idx]
    similar_games = sorted(similar_games, key=lambda x: x[1], reverse=True)

    # Obtiene los 5 juegos más similares
    similar_game_indices = [i[0] for i in similar_games[:5]]

    # Lista de juegos similares (solo nombres)
    similar_game_names = df_sample['app_name'].iloc[similar_game_indices].tolist()

    return {"similar_games": similar_game_names}