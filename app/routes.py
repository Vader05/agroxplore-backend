from flask import Blueprint, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import math
import os

# Obtener la ruta del directorio actual del script
path_base = os.path.dirname(os.path.abspath(__file__))

# loading models
temperature_min_model = load_model(os.path.join(path_base, 'models/temperatura_minima_a_2_metros/temperatura_minima_a_2_metros_model.h5'))
temperature_min_scaler_X = joblib.load(os.path.join(path_base, 'models/temperatura_minima_a_2_metros/temperatura_minima_a_2_metros_scaler_X.pkl'))
temperature_min_scaler_Y = joblib.load(os.path.join(path_base, 'models/temperatura_minima_a_2_metros/temperatura_minima_a_2_metros_scaler_Y.pkl'))

temperature_max_model = load_model(os.path.join(path_base,'models/temperature_at_2_meters_maximum/temperature_at_2_meters_maximum_model.h5'))
temperature_max_scaler_X = joblib.load(os.path.join(path_base, 'models/temperature_at_2_meters_maximum/temperature_at_2_meters_maximum_scaler_X.pkl'))
temperature_max_scaler_Y = joblib.load(os.path.join(path_base, 'models/temperature_at_2_meters_maximum/temperature_at_2_meters_maximum_scaler_Y.pkl'))

wind_speed_model = load_model(os.path.join(path_base, 'models/wind_speed_at_2_meters/wind_speed_at_2_meters_model.h5'))
wind_speed_scaler_X = joblib.load(os.path.join(path_base, 'models/wind_speed_at_2_meters/wind_speed_at_2_meters_X.pkl'))
wind_speed_scaler_Y = joblib.load(os.path.join(path_base, 'models/wind_speed_at_2_meters/wind_speed_at_2_meters_Y.pkl'))

specific_humidity_model = load_model(os.path.join(path_base, 'models/specific_humidity/specific_humidity_model.h5'))
specific_humidity_scaler_X = joblib.load(os.path.join(path_base, 'models/specific_humidity/specific_humidity_scaler_X.pkl'))
specific_humidity_scaler_Y = joblib.load(os.path.join(path_base, 'models/specific_humidity/specific_humidity_scaler_Y.pkl'))

precipitation_average_model = load_model(os.path.join(path_base, 'models/precipitation_average/precipitation_average_model.h5'))
precipitation_average_scaler_X = joblib.load(os.path.join(path_base, 'models/precipitation_average/precipitation_average_scaler_X.pkl'))
precipitation_average_scaler_Y = joblib.load(os.path.join(path_base, 'models/precipitation_average/precipitation_average_scaler_Y.pkl'))

bp = Blueprint('routes', __name__)

@bp.route('/predict/<param>', methods=['POST'])
def predict(param):
    data = request.get_json()
    lat = data['lat']
    lon = data['lon']
    year = data['year']
    
    # Crear la entrada para el modelo
    datos = pd.DataFrame({
        'LAT': [lat],
        'LON': [lon],
        'YEAR': [year]
    })

    input_data_cols = datos[['YEAR', 'LAT', 'LON']]
    prediction = []

    if(param == 'temperatureMin'):
        prediction = executeModel(temperature_min_scaler_Y, temperature_min_scaler_X, temperature_min_model, input_data_cols)
    
    if(param == 'temperatureMax'):
        prediction = executeModel(temperature_max_scaler_Y, temperature_max_scaler_X, temperature_max_model, input_data_cols)
    
    if(param == 'windSpeed'):
        prediction = executeModel(wind_speed_scaler_Y, wind_speed_scaler_X, wind_speed_model, input_data_cols)

    if(param == 'specificHumidity'):
        prediction = executeModel(specific_humidity_scaler_Y, specific_humidity_scaler_X, specific_humidity_model, input_data_cols)

    if(param == 'precipitationAverage'):
        prediction = executeModel(precipitation_average_scaler_Y, precipitation_average_scaler_X, precipitation_average_model, input_data_cols)

    if(len(prediction) == 0):
        return jsonify({"status": "error", "message": "There's not model to compute preduction for parameter " + param})

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    return jsonify({"status": "OK", "data": dict(zip(months, prediction.flatten().tolist()))})

def executeModel(scaler_y, scaler_x, model, data):
        # Normalizar la entrada (ajusta según tu método de normalización)
    input_data_scaled = scaler_x.transform(data) 

    # Realiza la predicción
    prediction_scaled = model.predict(input_data_scaled)
    
    # Inversa de la normalización (ajusta según tu scaler)
    return scaler_y.inverse_transform(prediction_scaled)

@bp.route('/calculate-wather', methods= ['GET'])
def calculateWather():
    product = request.args.get('product') # papa yungay, papa huayro
    area = request.args.get('area') #area de cultivo
    month = request.args.get('month') # jan, feb, mar

    area_hectareas = float(area) #hectareas

    # Leer el archivo CSV
    radiation = pd.read_csv(os.path.join(path_base, 'data/all_sky_surface_longwave_downward_irradiance.csv'))
    temperature = pd.read_csv(os.path.join(path_base, 'data/temperature_at_2_meters.csv'))
    wind_speed = pd.read_csv(os.path.join(path_base, 'data/wind_speed_at_2_meters.csv'))
    humedity_relative = pd.read_csv(os.path.join(path_base, 'data/humedity_relative.csv'))


    # Filtrar solo por parametros
    radiation_df = radiation[radiation['PARAMETER'] == 'ALLSKY_SFC_LW_DWN']
    temperature_df = temperature[temperature['PARAMETER'] == 'T2M']
    wind_speed_df = wind_speed[wind_speed['PARAMETER'] == 'WS2M']
    humedity_relative_df = humedity_relative[humedity_relative['PARAMETER'] == 'RH2M']

    meses = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    # Obtener el índice del mes inicial
    mes_index = meses.index(month)

    Kc_values = [0.97, 0.8, 0.33, 0.64 ,0.97 ,0.99] #constantes de kc

    # Generar la lista de meses seleccionados cíclicamente
    meses_seleccionados = [meses[(mes_index + i) % 12] for i in range(len(Kc_values))]

    print(meses_seleccionados)

    # Calcular el promedio de los meses seleccionados
    radiation_avg = radiation_df[meses_seleccionados].mean()
    temperature_avg = temperature_df[meses_seleccionados].mean()
    wind_speed_avg = wind_speed_df[meses_seleccionados].mean()
    humedity_relative_avg = humedity_relative_df[meses_seleccionados].mean()
    
    agua_necesaria_total = 0

    for month, Kc in enumerate(Kc_values, start=0):

        agua_necesaria = calcular_evapotranspiracion(temperature_avg[month], humedity_relative_avg[month], wind_speed_avg[month], radiation_avg[month]) * Kc * area_hectareas
        agua_necesaria_total += agua_necesaria
        print(f"Mes {month}: Agua necesaria = {agua_necesaria:.2f} mm")

    #print(f"\nEl agua total necesaria para el cultivo de papa Yungay en 7 meses es: {agua_necesaria_total:.2f} mm")


    return jsonify({"quantity":agua_necesaria_total, "product": product, "area": area})


def calcular_evapotranspiracion(T_promedio, RH_promedio, V_promedio, Rn_promedio):
    """
    Calcula la evapotranspiración (ET) usando la ecuación de Penman-Monteith.
    
    Parámetros:
    T_promedio : float : Temperatura media mensual (°C)
    RH_promedio : float : Humedad relativa media mensual (%) 
    V_promedio : float : Velocidad del viento media mensual (m/s)
    Rn_promedio : float : Radiación neta media mensual (MJ/m²/día)
    
    Retorna:
    ET : float : Evapotranspiración mensual (mm)
    """

    # Constantes
    delta = 4098 * (0.6108 * math.exp((17.27 * T_promedio) / (T_promedio + 237.3))) / (T_promedio + 237.3)**2  # kPa/°C
    gamma = 0.066  # kPa/°C

    # Calcular evapotranspiración (ET)
    ET = (0.408 * delta * Rn_promedio + gamma * (900 / (T_promedio + 273)) * V_promedio * (1 - RH_promedio / 100)) / (delta + gamma)

    return ET



