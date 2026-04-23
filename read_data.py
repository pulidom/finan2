'''
    Acceso a los datos a traves de load_ts. La primera vez se seleccionan las serie de datos completas y genera un npz  para que sea mas eficiente la lectura de los datos. Posteriores lecturas van a leer desde el npz.

    load_ts: Read an npz file with the time series to analyze
    csv2npz: Read  csv file, clean the data (select companies with whole time series and save in an npz

'''
import numpy as np, os
import pandas as pd
import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
try:
    import yfinance as yf
except Exception:  # pragma: no cover - opcional para flujos basados en CSV
    yf = None


def detect_raw_csv_sources(folder='./dat/'):
    candidates = [
        (folder + "tickers.csv", folder + "historical_prices.csv"),
        (folder + "tickers.csv", folder + "historical_prices-2.csv"),
        (folder + "tickers.csv", folder + "prices.csv"),
        (folder + "stock_metadata.csv", folder + "historical_prices.csv"),
        (folder + "stock_metadata.csv", folder + "historical_prices-2.csv"),
    ]
    for meta_file, price_file in candidates:
        if os.path.isfile(meta_file) and os.path.isfile(price_file):
            return meta_file, price_file
    raise Exception(f"El path: {folder} debe contener los archivos CSV correspondientes.")


def normalize_price_csv(df_dat):
    df_dat = df_dat.copy()
    if 'close_price' in df_dat.columns:
        df_dat = df_dat.rename(columns={
            'ticker_name': 'symbol',
            'close_price': 'close',
            'volume_in_units': 'volume',
            'open_price': 'open'
        })
    elif 'close' in df_dat.columns:
        rename_map = {}
        if 'ticker_name' in df_dat.columns and 'symbol' not in df_dat.columns:
            rename_map['ticker_name'] = 'symbol'
        if rename_map:
            df_dat = df_dat.rename(columns=rename_map)

    required_cols = {'date', 'symbol', 'close'}
    missing = required_cols.difference(df_dat.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en precios: {sorted(missing)}")

    if 'volume' not in df_dat.columns:
        df_dat['volume'] = np.nan

    return df_dat


def load_ts(assets=None,sectors=['oil'],pathdat='./dat/',
             init_date='2014-01-01',end_date='2024-12-31'):
    ''' Lee los datos ya sea un set especificado de assets o todos '''
    if sectors is None:
        sectors = list(sector_d.keys())

    prices, companies = [], []
    for sector in sectors:
        day,dates,price,company,volume = read_npz(sector=sector,pathdat=pathdat,
                                                  init_date=init_date,end_date=end_date)

        if assets is not None:
            prices=[]
            for asset in assets:
                #print(asset,company)
                j=np.where(company == asset)
                #print('aca',j)
                #print('shape: ',price.shape)
                if np.size(j) == 0:
                    print('El asset: ',asset,'no esta disponible en',company)
                    raise SystemExit
                else:
                    prices.append(price[j].squeeze())
                    companies.append(company[j])

            #prices=np.array(prices).T
        else:
            prices.append(price)
            companies.append(company)
            
    prices=np.array(prices).squeeze()
    companies = np.array(companies).squeeze()

    return day, dates, prices,companies,volume


def read_npz(sector='oil', pathdat='./dat/',
             init_date='2014-01-01',end_date='2024-12-31'):
    ''' Lee la series de datos en npz'''
    dat_fname = check_filename_exists(sector,pathdat,init_date,end_date)
    with np.load(dat_fname,allow_pickle=True) as dat:
        day = np.array(dat['day'])
        startdate = pd.to_datetime(str(dat['startdate'])).to_pydatetime()
        dates = np.array([startdate + datetime.timedelta(days=int(d)) for d in day])
        price = np.array(dat['price'])
        company = np.array(dat['company'])
        volume = np.array(dat['volume'])
    return day,dates,price,company,volume
    
def clean_data(day, price, company, volume, min_coverage=0.95):
    ''' Dado los prices en un periodo de tiempos 
    busca las compañias que tengan al menos `min_coverage` fraccion de datos completos.
    Los NaN restantes se rellenan con forward-fill. '''
    
    ncompany, nt = price.shape    
    nts = np.count_nonzero(~np.isnan(price[:, :]), axis=1)
    nt_correct = np.max(nts)
    jref = np.argmax(nts == nt_correct)
    nt_correct = np.count_nonzero(~np.isnan(price[jref, :]))
    mask_nan = np.logical_not(np.isnan(price[jref, :]))
    dt = day[jref, mask_nan]
    
    threshold = int(nt_correct * min_coverage)
    print(f'Dias habiles de referencia: {nt_correct} | Minimo requerido ({int(min_coverage*100)}%): {threshold}')

    prices, company1, volumes = [], [], []
    for i in range(ncompany):
        n_valid = np.count_nonzero(~np.isnan(price[i, mask_nan]))
        if n_valid >= threshold:
            p_slice = price[i, mask_nan].copy()
            v_slice = volume[i, mask_nan].copy()
            # Forward-fill NaNs remanentes
            df_tmp = pd.Series(p_slice)
            p_slice = df_tmp.ffill().bfill().values
            df_vol = pd.Series(v_slice)
            v_slice = df_vol.ffill().bfill().fillna(0).values
            prices.append(p_slice)
            volumes.append(v_slice)
            company1.append(company[i])

    price = np.array(prices)
    volume = np.array(volumes)
    company = np.array(company1)
    print(f'Cantidad de companias aceptadas: {len(company1)}')
    
    return dt, price, company, volume


def clean_data_on_common_calendar(day_offsets, price, company, volume, min_coverage=0.95):
    '''
    Limpia una matriz activo x tiempo preservando un calendario comun ya fijado.

    A diferencia de `clean_data`, no recorta fechas usando un ticker de referencia.
    Esto permite que todas las industrias compartan exactamente el mismo eje temporal.
    '''

    day_offsets = np.asarray(day_offsets, dtype=int)
    price = np.asarray(price, dtype=float)
    volume = np.asarray(volume, dtype=float)
    company = np.asarray(company)

    if price.ndim != 2:
        raise ValueError("`price` debe ser una matriz 2D de activos x fechas.")
    if volume.shape != price.shape:
        raise ValueError("`volume` debe tener la misma forma que `price`.")
    if day_offsets.ndim != 1 or len(day_offsets) != price.shape[1]:
        raise ValueError("`day_offsets` debe ser un vector 1D con una entrada por fecha del calendario comun.")

    nt = len(day_offsets)
    threshold = int(np.ceil(nt * float(min_coverage)))
    print(f'Fechas del calendario comun: {nt} | Minimo requerido ({int(min_coverage*100)}%): {threshold}')

    prices, company1, volumes = [], [], []
    for i in range(price.shape[0]):
        valid_mask = ~np.isnan(price[i, :])
        n_valid = int(np.count_nonzero(valid_mask))
        if n_valid < threshold:
            continue

        p_slice = pd.Series(price[i, :]).ffill().bfill().values
        v_slice = pd.Series(volume[i, :]).ffill().bfill().fillna(0).values
        prices.append(p_slice)
        volumes.append(v_slice)
        company1.append(company[i])

    price = np.array(prices)
    volume = np.array(volumes)
    company = np.array(company1)
    print(f'Cantidad de companias aceptadas en calendario comun: {len(company1)}')

    return day_offsets, price, company, volume


def check_filename_exists(sector,pathdat,init_date,end_date):
    ''' Chequea si npz-file existe sino llama a cvs2npz
       y lo genera '''
    full_fname=f'{pathdat}/{sector}_{init_date}_{end_date}.npz'
    if not os.path.isfile(full_fname):
        # Get the sector
        csv2npz(init_date=init_date,end_date=end_date,
                folder=pathdat,
                industry_type=sector,
                fname=full_fname,
            )
    return full_fname

sector_d = {"airlines":"Passenger Airlines", 
            "construction": "Construction & Engineering", 
            "biotechnology": "Biotechnology", 
            "defense": "Aerospace & Defense" , 
            "hotels": "Hotels, Restaurants & Leisure",
            "insurance": "Insurance",
            "marine": "Marine Transportation",
            "metals": "Metals & Mining",
            "oil": "Energy - Fossil Fuels",
            "banks": "Banking Services",
            "semiconductors": "Semiconductors & Semiconductor Equipment",
            "software": "Software",
            "pharmaceuticals": "Pharmaceuticals",
            "chemicals": "Chemicals",
            "capital": "Capital Markets",
            "containers": "Containers & Packaging",
            "water": "Water Utilities",
            "machinery": "Machinery",
            "beverages": "Beverages",
            "media": "Media",
            "interactive": "Interactive Media & Services",
            "automobiles": "Automobiles",
            "hardware": "Technology Hardware, Storage & Peripherals",
            "broadline": "Broadline Retail"
          }

def csv2npz(init_date='2014-01-01',end_date='2024-12-31',
            var_type='close',
            folder='./dat/',
            fname=None,
            industry_type='oil'):
    ''' Dado un periodo de tiempos y una industria lee el csv y 
       filtra las compa~nias que tienen datos completos en el periodo
       y los guarda en un npz '''

    sector = sector_d.get(industry_type, industry_type)
    
    # Identificar nombres de archivo según disponibilidad
    meta_file, price_file = detect_raw_csv_sources(folder)
        
    # reading csv file 
    df = pd.read_csv(meta_file, low_memory=False)
    df_dat = normalize_price_csv(pd.read_csv(price_file, low_memory=False))

    # Normalización de columnas para estandarización
    if 'RIC' in df.columns:
        df = df.rename(columns={'RIC': 'symbol'})
    elif 'ticker_name' in df.columns:
        df = df.rename(columns={'ticker_name': 'symbol'})

    mask = pd.Series(False, index=df.index)
    col_sectores = ['TRBC Business Sector', 'TRBC Industry Group', 'industry', 'Sector', 'TRBC Economic Sector']
    for col in col_sectores:
        if col in df.columns:
            mask = mask | (df[col] == sector)
            
    if mask.any() or any(c in df.columns for c in col_sectores):
        df_company = df[mask]
    else:
        df_company = df

    df_dat['date'] = pd.to_datetime(df_dat['date'], errors='coerce')
    df_dat = df_dat.dropna(subset=['date'])

    print('Finished reading csv')
    init_date=pd.to_datetime(init_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=init_date, end=end_date, freq='D')

    julians, opens, company, volumes = [], [], [], []
    df_all = pd.DataFrame({'date': date_range})

    print('Collecting time series')
    # Optimización: filtrar y pre-agrupar para evitar O(N*M)
    df_dat = df_dat[(df_dat['date'] >= init_date) & (df_dat['date'] <= end_date)].copy()
    df_dat.loc[:, 'julian'] = (df_dat['date'] - init_date).dt.days
    df_dat_grouped = dict(tuple(df_dat.groupby('symbol')))

    for index, row in df_company.iterrows():
        sym = row['symbol']
        if sym not in df_dat_grouped:
            continue
        df_ts = df_dat_grouped[sym]

        df_rangets=pd.merge(df_all, df_ts[['date', 'julian', var_type,'volume']], on='date', how='left')

        julians.append(df_rangets['julian'].values)
        opens.append(df_rangets[var_type].values)
        volumes.append(df_rangets['volume'].values)
        company.append(sym)

    day = np.array(julians)
    price = np.array(opens)
    volume = np.array(volumes)
    company = np.array(company)
    print('Before cleaning: ',price.shape)
    dt,price,company,volume = clean_data(day,price,company,volume)


    print('After cleaning: ',price.shape)
    if fname is None:
        fname=f'{folder}/{sector}_{init_date}_{end_date}.npz'
    np.savez(fname,
             day=dt,price=price,volume=volume, company=company, startdate=init_date)

def yahoo_download(tickers, start_date, end_date):
    """
    Descarga datos financieros de Yahoo Finance y calcula retornos
    
    Args:
        tickers: lista de símbolos de acciones/ETFs
        start_date: fecha de inicio (YYYY-MM-DD)
        end_date: fecha de fin (YYYY-MM-DD)
        return_type: tipo de retorno a calcular ('price', 'log', 'return')
            - 'price': precios ajustados sin transformación
            - 'log': retornos logarítmicos
            - 'return': retornos simples (porcentuales)
    
    Returns:
        returns_df: DataFrame con retornos/precios según return_type
        prices_df: DataFrame con precios ajustados
        volumes_df: DataFrame con volúmenes operados en dólares
    """
    if yf is None:
        raise ImportError("yfinance no está instalado en este entorno.")
    print(f"Periodo: {start_date} a {end_date}")

    
    data = yf.download(tickers, start=start_date, end=end_date,
                        progress=False,auto_adjust=True)
    
    if len(tickers) == 1:
        # Para un solo ticker, yfinance puede devolver diferentes estructuras
        if 'Close' in data.columns:
            prices_df = data['Close'].to_frame()
            prices_df.columns = tickers
        else:
            # Si no hay columna 'Adj Close', usar la primera columna disponible
            prices_df = data.iloc[:, 0].to_frame()
            prices_df.columns = tickers
            #print("Warning: 'Adj Close' no encontrado, usando primera columna disponible")
        
        # Obtener volúmenes
        if 'Volume' in data.columns:
            volumes_df = data['Volume'].to_frame()
            volumes_df.columns = tickers
        else:
            volumes_df = pd.DataFrame(index=data.index, columns=tickers)
            print("Warning: 'Volume' no encontrado")
    else:
        # Para múltiples tickers
        import pandas as pd
        if isinstance(data.columns, pd.MultiIndex):
            
            prices_df = data['Close']
            if 'Volume' in data.columns.get_level_values(0):
                volumes_df = data['Volume']
            else:
                volumes_df = pd.DataFrame(index=data.index, columns=tickers)
                print("Warning: 'Volume' no encontrado")
        else:
            # Estructura simple de columnas
            prices_df = data
            volumes_df = pd.DataFrame(index=data.index, columns=tickers)
            print("Warning: Estructura de datos simple, volúmenes no disponibles")
    
    # Verificar que tenemos al menos algunos datos válidos
    valid_tickers = []
    for ticker in tickers:
        if ticker in prices_df.columns:
            if not prices_df[ticker].isna().all():
                valid_tickers.append(ticker)
    
    if not valid_tickers:
        raise ValueError("No se encontraron datos válidos para ningún ticker")
    
    # Filtrar solo los tickers válidos
    prices_df = prices_df[valid_tickers]
    
    # Filtrar volúmenes para los tickers válidos
    volumes_df = volumes_df[valid_tickers] if all(ticker in volumes_df.columns for ticker in valid_tickers) else pd.DataFrame(index=prices_df.index, columns=valid_tickers)
    
    # Calcular volúmenes en dólares (Volume * Precio de cierre ajustado)
    dollar_volumes_df = volumes_df.copy()
    for ticker in valid_tickers:
        if ticker in volumes_df.columns and ticker in prices_df.columns:
            dollar_volumes_df[ticker] = volumes_df[ticker] * prices_df[ticker]
        else:
            print(f"Warning: No se pudo calcular volumen en dólares para {ticker}")
    
    day   = np.arange(1,len(prices_df))
    date  = pd.date_range(start=start_date, end=end_date, freq="D").to_numpy()
    price = prices_df.to_numpy()
    company= np.array([tickers]).T[:,0]
    #return returns_df, prices_df, dollar_volumes_df
    return day,date,price.T,company,dollar_volumes_df.to_numpy().T
    
if __name__=="__main__":
    csv2npz(init_date='2014-01-01',end_date='2024-12-31')
