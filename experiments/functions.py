import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def PlotPredictions(plots,title):
    plt.figure(figsize=(18, 8))
    for plot in plots:
        plt.plot(plot[0], plot[1], label=plot[2], linestyle=plot[3], color=plot[4],lw=1)
    plt.xlabel('Date')
    plt.ylabel("Sales")
    plt.title(title)
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    plt.show()
def CalculateError(pred,sales):
    percentual_errors = []
    for A_i, B_i in zip(sales, pred):
        percentual_error = abs((A_i - B_i) / B_i)
        percentual_errors.append(percentual_error)
    return sum(percentual_errors) / len(percentual_errors)

def create_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week  # Updated line

    X = df[['hour', 'dayofweek', 'quarter', 'month', 'year',
            'dayofyear', 'dayofmonth', 'weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
def create_lagged_features(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
def CalculateError(pred,sales):
    percentual_errors = []
    for A_i, B_i in zip(sales, pred):
        percentual_error = abs((A_i - B_i) / B_i)
        percentual_errors.append(percentual_error)
    return sum(percentual_errors) / len(percentual_errors)
# Función para crear las características y etiquetas de las series temporales
def create_lstm_features(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])  # Ventana de características
        y.append(data[i, 0])  # Siguiente valor como etiqueta
    return np.array(X), np.array(y)
def PlotGraph(sales, data_column, filepath=None):
    # Configurar el estilo de matplotlib
    plt.style.use('seaborn-v0_8-darkgrid')

    # Configurar el tamaño de la figura
    plt.figure(figsize=(16, 8))

    # Graficar los datos con una línea más gruesa y color atractivo
    plt.plot(sales[data_column], linewidth=1, color='teal', label='Ventas')

    # Configurar los ticks del eje X para mostrar solo un punto cada año
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())  # Un tick cada año
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # Mostrar solo el año

    # Rotar las etiquetas del eje X
    plt.xticks(rotation=45, ha='right')

    # Añadir título y etiquetas
    plt.title('Evolución de las Ventas en el Tiempo', fontsize=18, fontweight='bold')
    plt.xlabel('Año', fontsize=14)
    plt.ylabel('Ventas', fontsize=14)

    # Añadir cuadrícula para facilitar la visualización
    plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

    # Añadir leyenda
    plt.legend(fontsize=12)

    # Mostrar el gráfico
    plt.tight_layout()
    if filepath:
        plt.savefig(filepath)
    plt.show()
