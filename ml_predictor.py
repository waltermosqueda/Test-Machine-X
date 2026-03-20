import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, precision_score
import xgboost as xgb
import warnings
import sys
import os
import gc
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACIÓN Y LISTA DE ACTIVOS
# ==============================================================================

activos = [
    'AAL', 'AAP', 'AAPL', 'ABBV', 'ABEV', 'ABT', 'ACN', 'ADBE', 'ADI', 'ADP',
    'AEG', 'AEM', 'AGRO', 'AIG', 'AMAT', 'AMD', 'AMGN', 'AMX', 'AMZN',
    'ANF', 'ARCO', 'ARM', 'ASR', 'AVGO', 'AVY', 'AXP', 'AZN', 'BA',
    'BABA', 'BAC', 'BAK', 'BB', 'BBD', 'BBVA', 'BG', 'BHP', 'BIDU',
    'BIIB', 'BK', 'BKNG', 'BKR', 'BMY', 'BP', 'BSBR', 'C',
    'CAAP', 'CAH', 'CAR', 'CAT', 'CCL', 'CDE', 'CL', 'COIN',
    'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'CX', 'DAL', 'DD', 'DE',
    'DEO', 'DHR', 'DIS', 'DOCU', 'DOW', 'E', 'EA', 'EBAY',
    'EFX', 'EQNR', 'ERIC', 'ETSY', 'FCX', 'FDX', 'FMX',
    'FSLR', 'GE', 'GFI', 'GGB', 'GILD', 'GLOB', 'GLW', 'GM', 'GOLD',
    'GOOGL', 'GPRK', 'GRMN', 'GS', 'GSK', 'GT', 'HAL', 'HD', 'HDB', 'HL',
    'HMC', 'HMY', 'HNHPF', 'HOG', 'HON', 'HPQ', 'HSBC', 'HSY', 'HWM', 'IBM',
    'IBN', 'IFF', 'INFY', 'ING', 'INTC', 'IP', 'ISRG', 'ITUB', 'JCI', 'JD',
    'JNJ', 'JPM', 'KB', 'KEP', 'KGC', 'KMB', 'KO', 'KOF', 'LAC', 'LAR',
    'LLY', 'LMT', 'LND', 'LRCX', 'LVS', 'LYG', 'MA',
    'MCD', 'MDLZ', 'MDT', 'MELI', 'META', 'MFG', 'MMM', 'MO', 'MRK',
    'MRNA', 'MRVL', 'MSFT', 'MSI', 'MUFG', 'MUX', 'NEM', 'NFLX', 'NG', 'NGG',
    'NIO', 'NKE', 'NMR', 'NOK', 'NSANY', 'NTES', 'NU', 'NUE', 'NVDA',
    'NVS', 'NXE', 'OAOFY', 'ORANY', 'ORCL', 'ORLY', 'PAAS', 'PAC',
    'PAGS', 'PBI', 'PBR', 'PCAR', 'PEP', 'PFE', 'PG', 'PHG', 'PINS',
    'PKX', 'PLTR', 'PM', 'PSO', 'PSX', 'PYPL', 'QCOM', 'RACE', 'RIO',
    'RIOT', 'ROKU', 'ROST', 'RTX', 'SAN', 'SAP', 'SBS', 'SBUX', 'SCCO', 'SCHW',
    'SDA', 'SE', 'SHEL', 'SHOP', 'SHPWQ', 'SID', 'SIEGY', 'SLB', 'SNA', 'SNAP',
    'SNOW', 'SONY', 'SPCE', 'SPGI', 'SPOT', 'STLA', 'STNE', 'SUZ', 'SWKS',
    'SYY', 'T', 'TCOM', 'TEF', 'TGT', 'TIIAY', 'TIMB', 'TJX', 'TMO',
    'TMUS', 'TRIP', 'TRV', 'TS', 'TSLA', 'TSM', 'TTE', 'TV', 'TWLO', 'TX',
    'TXN', 'UGP', 'UL', 'UNH', 'UNP', 'URBN', 'USB', 'V', 'VALE', 'VIST',
    'VIV', 'VOD', 'VRSN', 'VZ', 'WB', 'WFC', 'WMT', 'XOM',
    'XP', 'XRX', 'XYZ', 'YELP', 'YZCAY', 'ZM'
]

VENTANA_PREDICCION = 5
UMBRAL_SUBA = 0.02
PERIODO_DATOS = '6mo'  # Reducido para ahorrar memoria

def calcular_indicadores(df):
    df = df.copy()
    for window in [5, 10, 20]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    df['SMA_Cross'] = df['SMA_5'] - df['SMA_20']
    df['EMA_Cross'] = df['EMA_5'] - df['EMA_20']
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df['ROC'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
    
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    for i in range(1, 4):
        df[f'Return_Lag_{i}'] = df['Close'].pct_change().shift(i)
        
    return df

def crear_target(df, ventana, umbral):
    futuro_max = df['Close'].shift(-ventana).rolling(window=ventana).max()
    retorno_futuro = (futuro_max - df['Close']) / df['Close']
    df['Target'] = (retorno_futuro >= umbral).astype(int)
    return df

# ==============================================================================
# PROCESAMIENTO PRINCIPAL
# ==============================================================================

print("Iniciando descarga y procesamiento de datos...")
sys.stdout.flush()
datos_combinados = []

for i, ticker in enumerate(activos):
    try:
        data = yf.download(ticker, period=PERIODO_DATOS, progress=False, timeout=5)
        
        if len(data) < 60:
            continue
            
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        data = calcular_indicadores(data)
        data = crear_target(data, VENTANA_PREDICCION, UMBRAL_SUBA)
        
        data['Ticker'] = ticker
        datos_combinados.append(data)
        
        if (i + 1) % 50 == 0:
            print(f"Procesados {i+1}/{len(activos)} activos...")
            sys.stdout.flush()
            gc.collect()
        
    except Exception as e:
        pass

if not datos_combinados:
    raise ValueError("No se pudieron descargar datos.")

df_total = pd.concat(datos_combinados, ignore_index=True)
del datos_combinados
gc.collect()

df_total = df_total.dropna()

feature_cols = [col for col in df_total.columns if col not in ['Target', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

X = df_total[feature_cols]
y = df_total['Target']

print(f"Entrenando modelo con {len(feature_cols)} características y {len(X)} muestras...")
sys.stdout.flush()

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]) if 1 in y_train.value_counts() else 1,
    eval_metric='auc',
    tree_method='hist',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- RESULTADOS DE VALIDACIÓN ---")
print(classification_report(y_test, y_pred, target_names=['No Sube', 'Sube']))
sys.stdout.flush()

print("\n--- PREDICCIONES ACTUALES ---")
ultimos_datos = df_total.groupby('Ticker').last().reset_index()

predicciones_finales = []
for index, row in ultimos_datos.iterrows():
    ticker = row['Ticker']
    features = row[feature_cols].values.reshape(1, -1).astype(np.float64)
    
    if np.isnan(features).any():
        continue
        
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]
    
    if pred == 1:
        predicciones_finales.append({
            'Ticker': ticker,
            'Probabilidad_Suba': prob,
            'Señal': 'COMPRA FUERTE' if prob > 0.7 else 'COMPRA MODERADA'
        })

if predicciones_finales:
    df_predicciones = pd.DataFrame(predicciones_finales).sort_values(by='Probabilidad_Suba', ascending=False)
    print(df_predicciones.head(20).to_string(index=False))
else:
    print("No se detectaron oportunidades claras hoy.")

print("\nProceso finalizado.")
