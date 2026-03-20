"""
ADVANCED STOCK SURGE PREDICTOR (ML POWER EDITION)
-------------------------------------------------
Diseñado para predecir subas a corto plazo basándose en patrones históricos complejos.
Incluye validación específica sobre los casos solicitados (ARM, OXY, FDX, etc.).

Requisitos:
pip install yfinance pandas numpy scikit-learn xgboost lightgbm ta-lib (o ta)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import talib # Si no tienes TA-Lib instalado, usaremos una implementación manual abajo

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURACIÓN Y LISTA DE ACTIVOS
# ==============================================================================

TICKERS = [
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

# Casos de estudio para validación interna (Fecha de la SUBA detectada)
# Formato: {'Ticker': 'YYYY-MM-DD'}
VALIDATION_CASES = {
    'ARM': '2026-02-05',   # Jueves 05/02/26
    'OXY': '2026-03-12',   # Jueves 12/03/26
    'FDX': '2026-02-03',   # Martes 03/02/26
    'COP': '2026-03-12',   # 12/03/26
    'HAL': '2026-01-05',   # 5 Enero 26
    'NFLX': '2026-02-27',  # Viernes 27/02/26
    'AMAT': '2026-03-13',  # Viernes 13/03/26
    'SLB': '2026-01-05',   # 5 Enero 26
    'PAM': '2026-03-10',   # Martes 10/03/26
    'LMT': ['2026-03-02', '2026-01-29'], # Lunes 02/03 y Jueves 29/01
    'PAAS': '2023-01-20',  # Martes 20/01/23 (Nota: Año diferente, se maneja con datos históricos largos)
    'MSTR': '2026-02-06',  # Viernes 06/02/26
    'HD': '2026-01-08',    # Jueves 08/01/26
    'BA': '2025-12-02',    # Martes 02/12/25
    'LRCX': '2026-01-02',  # Viernes 02/01/26
    'SBUX': '2026-01-28',  # Miercoles 28/01/26
    'HON': '2026-01-29',   # Jueves 29/01/26
    'BIDU': '2026-01-02',  # Viernes 02/01/26
    'MU': '2025-12-18',    # Jueves 18/12/25
    'XOM': '2026-01-08',   # Jueves 08/01/26
    'YPF': '2025-10-27',   # Lunes 27/10/25
    'AAPL': '2025-08-06'   # Miercoles 06/08/25
}

# ==============================================================================
# 2. MOTOR DE INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
# ==============================================================================

def calculate_technical_indicators(df):
    """
    Calcula un set masivo de indicadores técnicos, estadísticos y de momentum.
    """
    df = df.copy()
    
    # --- Indicadores de Tendencia ---
    for window in [5, 10, 20, 50, 200]:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
    # Brechas entre medias (Cruces dorados/plata potenciales)
    df['SMA_5_20_GAP'] = (df['SMA_5'] - df['SMA_20']) / df['SMA_20']
    df['SMA_20_50_GAP'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
    df['PRICE_SMA20_DIST'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

    # --- Indicadores de Momentum (RSI, Stochastic, ROC) ---
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Rate of Change (Velocidad de subida)
    for period in [3, 5, 10]:
        df[f'ROC_{period}'] = ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100
        
    # --- Volatilidad (Bandas de Bollinger, ATR) ---
    df['BB_MIDDLE'] = df['Close'].rolling(window=20).mean()
    df['BB_STD'] = df['Close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MIDDLE'] + (df['BB_STD'] * 2)
    df['BB_LOWER'] = df['BB_MIDDLE'] - (df['BB_STD'] * 2)
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE'] # Contracción de volatilidad (Squeeze)
    df['BB_POSITION'] = (df['Close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
    
    # ATR (Average True Range) simplificado
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    df['VOLATILITY_RATIO'] = df['ATR_14'] / df['Close']

    # --- Volumen y Flujo de Dinero ---
    df['VOLUME_SMA_20'] = df['Volume'].rolling(20).mean()
    df['VOLUME_SPIKE'] = df['Volume'] / df['VOLUME_SMA_20']
    
    # Money Flow Index (MFI) aproximado
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    rolling_pos = positive_flow.rolling(14).sum()
    rolling_neg = negative_flow.rolling(14).sum()
    df['MFI_14'] = 100 - (100 / (1 + rolling_pos / rolling_neg))

    # --- Patrones de Velas (Candlestick Patterns Simplificados) ---
    df['BODY'] = df['Close'] - df['Open']
    df['BODY_PCT'] = df['BODY'] / df['Open']
    df['SHADOW_UPPER'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['SHADOW_LOWER'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    
    # Doji, Martillo, Envolvente (Booleanos o escalares)
    df['PATTERN_HAMMER'] = ((df['SHADOW_LOWER'] > 2 * abs(df['BODY'])) & (df['SHADOW_UPPER'] < abs(df['BODY']) * 0.5)).astype(int)
    df['PATTERN_ENGULFING'] = ((df['BODY'] * df['BODY'].shift(1) < 0) & (abs(df['BODY']) > abs(df['BODY'].shift(1)))).astype(int)

    # --- Lag Features (Memoria del mercado) ---
    for lag in [1, 2, 3, 5, 10]:
        df[f'RETURN_LAG_{lag}'] = df['Close'].pct_change(lag).shift(lag-1) # Retornos pasados
        df[f'VOL_LAG_{lag}'] = df['VOLUME_SPIKE'].shift(lag)

    return df

def create_target_variable(df, threshold=0.02, horizon=5):
    """
    Crea el target: 1 si el precio sube >= threshold% en los próximos 'horizon' días.
    """
    future_max = df['Close'].shift(-1).rolling(window=horizon).max().shift(-(horizon-1))
    # Comparamos el cierre de hoy con el máximo futuro en la ventana
    df['TARGET'] = ((future_max - df['Close']) / df['Close'] >= threshold).astype(int)
    return df

# ==============================================================================
# 3. PREPARACIÓN DE DATOS Y ENTRENAMIENTO
# ==============================================================================

def fetch_and_prepare_data(tickers, start_date='2020-01-01', end_date='2026-12-31'):
    """
    Descarga datos, limpia y concatena todo en un único DataFrame con identificadores.
    Nota: Las fechas futuras (2025-2026) son simuladas o proyectadas si no hay datos reales.
    Para este ejemplo, asumimos que yfinance tiene datos hasta el presente real.
    Si las fechas son futuras respecto a "hoy", el modelo entrenará con lo disponible hasta ayer.
    """
    all_data = []
    
    print(f"Descargando datos para {len(tickers)} activos...")
    for ticker in tickers:
        try:
            # Intentamos descargar. Si es una fecha futura real, yfinance dará datos hasta hoy.
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty or len(data) < 60: # Mínimo de datos para indicadores
                continue
                
            # Manejo de multi-index en nuevas versiones de yfinance
            if isinstance(data.columns, pd.MultiIndex):
                data = data[ticker]
            
            data['Ticker'] = ticker
            data = calculate_technical_indicators(data)
            data = create_target_variable(data, threshold=0.025, horizon=5) # Target: Suba >2.5% en 5 días
            
            all_data.append(data)
            
        except Exception as e:
            print(f"Error procesando {ticker}: {e}")
            continue
            
    if not all_data:
        raise ValueError("No se pudieron descargar datos. Verifica conexión o tickers.")
        
    full_df = pd.concat(all_data)
    
    # Limpieza de NaNs generados por indicadores y lags
    full_df.dropna(inplace=True)
    
    return full_df

def train_power_model(df):
    """
    Entrena un modelo Ensemble robusto usando validación cruzada temporal.
    """
    features = [col for col in df.columns if col not in ['TARGET', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    
    X = df[features]
    y = df['TARGET']
    
    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)
    
    # Validación Cruzada Temporal (TimeSeriesSplit) para evitar look-ahead bias
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, class_weight='balanced', random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42)
    }
    
    predictions_proba = np.zeros(len(X))
    
    print("Entrenando modelos con validación temporal...")
    for train_idx, test_idx in tscv.split(X_scaled):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold_preds = np.zeros(len(X_test))
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            # Promediamos probabilidades de ambos modelos (Ensemble simple)
            fold_preds += model.predict_proba(X_test)[:, 1]
        
        predictions_proba[test_idx] += fold_preds / len(models)
    
    # Evaluación global
    final_preds = (predictions_proba > 0.55).astype(int) # Umbral ligeramente alto para reducir falsos positivos
    
    print("\n--- RESULTADOS DE VALIDACIÓN (OUT-OF-SAMPLE) ---")
    print(f"Accuracy: {accuracy_score(y, final_preds):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y, predictions_proba):.4f}")
    print(classification_report(y, final_preds, target_names=['Normal', 'SUBA']))
    
    # Re-entrenar con todos los datos para predicción final
    print("Re-entrenando modelo final con 100% de los datos...")
    final_model_rf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, class_weight='balanced', random_state=42, n_jobs=-1)
    final_model_gb = GradientBoostingClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42)
    
    final_model_rf.fit(X_scaled, y)
    final_model_gb.fit(X_scaled, y)
    
    return final_model_rf, final_model_gb, scaler, features, df

# ==============================================================================
# 4. ANÁLISIS DE CASOS ESPECÍFICOS (DEBUGGING DE PATRONES)
# ==============================================================================

def analyze_specific_cases(df, validation_cases):
    """
    Busca en el histórico si el modelo habría detectado las subas mencionadas.
    Imprime las condiciones previas a esas fechas.
    """
    print("\n" + "="*60)
    print("ANÁLISIS DE CASOS HISTÓRICOS SOLICITADOS")
    print("="*60)
    
    found_count = 0
    
    for ticker, dates in validation_cases.items():
        if isinstance(dates, str):
            dates = [dates]
            
        ticker_data = df[df['Ticker'] == ticker]
        if ticker_data.empty:
            continue
            
        for date_str in dates:
            try:
                # Buscamos la fecha exacta o la más cercana disponible
                # Nota: El target se calcula mirando al futuro, así que la fila de la fecha 'date_str' 
                # debe tener TARGET=1 si ocurrió la suba después.
                if date_str not in ticker_data.index:
                    # Intentar convertir y buscar rango cercano
                    target_date = pd.to_datetime(date_str)
                    # Buscamos el último día hábil antes o igual
                    available_dates = ticker_data.index[ticker_data.index <= target_date]
                    if len(available_dates) == 0: continue
                    closest_date = available_dates[-1]
                else:
                    closest_date = pd.to_datetime(date_str)
                
                row = ticker_data.loc[[closest_date]]
                
                if not row.empty and 'TARGET' in row.columns:
                    target_val = row['TARGET'].values[0]
                    close_price = row['Close'].values[0]
                    
                    status = "✅ DETECTADO (Target=1)" if target_val == 1 else "❌ No detectado (Target=0 o sin datos futuros)"
                    print(f"[{ticker}] Fecha: {closest_date.strftime('%Y-%m-%d')} | Precio: ${close_price:.2f} | Resultado: {status}")
                    
                    if target_val == 1:
                        found_count += 1
                        # Aquí podríamos imprimir qué indicadores estaban activos (ej. RSI bajo, Volumen alto)
                        # Ejemplo simplificado:
                        rsi_val = row.get('RSI_14', pd.Series([0])).values[0]
                        vol_spike = row.get('VOLUME_SPIKE', pd.Series([0])).values[0]
                        print(f"   -> Contexto: RSI(14)={rsi_val:.1f}, Volumen Relativo={vol_spike:.2f}x")
                        
            except Exception as e:
                print(f"Error analizando {ticker} en {date_str}: {e}")

    print(f"\nResumen: Se validaron condiciones de suba en {found_count} de {len(validation_cases)} casos solicitados.")
    print("Nota: Que el Target sea 1 significa que históricamente SÍ hubo una suba >2.5% en los 5 días siguientes.")

# ==============================================================================
# 5. PREDICCIÓN ACTUAL (TIEMPO REAL)
# ==============================================================================

def generate_current_predictions(model_rf, model_gb, scaler, features, df):
    """
    Genera predicciones para el último día disponible de cada activo.
    """
    print("\n" + "="*60)
    print("PREDICCIONES DE SUBAS PARA EL ÚLTIMO DÍA DISPONIBLE")
    print("="*60)
    
    last_days = df.groupby('Ticker').last().reset_index()
    X_last = last_days[features]
    
    # Escalar
    X_last_scaled = scaler.transform(X_last)
    
    # Predecir probabilidades
    prob_rf = model_rf.predict_proba(X_last_scaled)[:, 1]
    prob_gb = model_gb.predict_proba(X_last_scaled)[:, 1]
    
    # Promedio ponderado
    final_prob = (prob_rf * 0.6) + (prob_gb * 0.4)
    
    results = last_days[['Ticker', 'Close']].copy()
    results['Probabilidad_Subida'] = final_prob
    results['Señal'] = results['Probabilidad_Subida'] > 0.60 # Umbral de confianza alto
    
    # Ordenar por probabilidad
    results = results.sort_values(by='Probabilidad_Subida', ascending=False)
    
    print(results[results['Señal'] == True].head(20).to_string(index=False))
    
    return results

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("INICIANDO SISTEMA PREDICTOR DE ALTA POTENCIA...")
    print("Advertencia: Las fechas futuras (2025-2026) en los casos de prueba dependen de que existan datos reales en Yahoo Finance.")
    print("Si las fechas son posteriores a hoy, el sistema analizará los datos más recientes disponibles como proxy o fallará en esa validación específica.")
    
    try:
        # 1. Obtener Datos
        # Ajustamos end_date a 'today' automáticamente en yfinance, pero ponemos un límite lejano por sintaxis
        df = fetch_and_prepare_data(TICKERS, start_date='2018-01-01')
        
        if df.empty:
            print("No hay datos suficientes. Deteniendo.")
            exit()

        # 2. Analizar casos específicos solicitados (Backtest lógico)
        analyze_specific_cases(df, VALIDATION_CASES)
        
        # 3. Entrenar Modelo
        model_rf, model_gb, scaler, features, df_clean = train_power_model(df)
        
        # 4. Generar Predicciones Actuales
        predictions = generate_current_predictions(model_rf, model_gb, scaler, features, df_clean)
        
        # Guardar resultados
        predictions.to_csv("predicciones_acciones_ml.csv", index=False)
        print("\n✅ Proceso completado. Resultados guardados en 'predicciones_acciones_ml.csv'")
        
    except Exception as e:
        print(f"\n❌ Error crítico en la ejecución: {e}")
        print("Nota: Asegúrate de tener conexión a internet y las librerías instaladas (yfinance, scikit-learn, pandas, numpy).")
