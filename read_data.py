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

def load_ts(assets=None,sector='oil',pathdat='./dat/',
             init_date='2014-01-01',end_date='2024-12-31'):
    ''' Lee los datos ya sea un set especificado de assets o todos '''
    day,dates,price,company,volume = read_npz(sector=sector,pathdat=pathdat,
                                              init_date=init_date,end_date=end_date)

    if assets is not None:
        prices=[]
        for asset in assets:
            j=np.where(company == asset)
            print('aca',j)
            print('shape: ',price[j].shape)
            prices.append(price[j].squeeze())
            
        prices=np.array(prices).T
    else:
        prices=price
        
    return day, dates, prices,company,volume

def read_npz(sector='oil', pathdat='./dat/',
             init_date='2014-01-01',end_date='2024-12-31'):
    ''' Lee la series de datos en npz'''
    dat_fname = check_filename_exists(sector,pathdat,init_date,end_date)
    dat=np.load(dat_fname,allow_pickle=True)    
    dates = np.array([dat['startdate'] + datetime.timedelta(days=int(d)) for d in dat['day']])
    return dat['day'],dates,dat['price'],dat['company'],dat['volume']
    
def clean_data(day,price,company,volume): 
    ''' Dado los prices en un periodo de tiempos 
    busca las compa~nias que tengan toda la series completa '''
    
    ncompany,nt = price.shape    
    nts = np.count_nonzero(~np.isnan(price[:,:]),axis=1)
    nt_correct=np.max(nts)
    jref=np.argmax(nts==nt_correct)
    nt_correct = np.count_nonzero(~np.isnan(price[jref,:]))
    mask_nan=np.logical_not(np.isnan(price[jref,:]))
    dt = day[jref,mask_nan]
    
    print('Dias habiles: ',nt_correct)

    prices,company1, volumes =  [], [], [] #np.zeros(price.shape[0],nt_correct)
    for i in range(ncompany):
        if nt_correct == np.count_nonzero(~np.isnan(price[i,:])):
            prices.append( price[i,mask_nan] )
            volumes.append( volume[i,mask_nan] )
            company1.append(company[i])
                            
    price = np.array(prices)
    volume = np.array(volumes)
    company = np.array(company1)
    print('Cantidad de compa~nias: ',len(company1))
    
    return dt,price,company,volume


def check_filename_exists(sector,pathdat,init_date,end_date):
    ''' Chequea si npz-file existe sino llama a cvs2npz
       y lo genera '''
    full_fname=f'{pathdat}/{sector}_{init_date}_{end_date}_day_vol_closeval.npz'
    if not os.path.isfile(full_fname):
        # Get the sector
        csv2npz(init_date=init_date,end_date=end_date,
                folder=pathdat,
                industry_type=sector,
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
            "oil": "Oil, Gas & Consumable Fuels" ,
            "semiconductors": "Semiconductors & Semiconductor Equipment",
            "software": "Software",
            "pharmaceuticals": "Pharmaceuticals",
            "chemicals": "Chemicals",
            "capital": "Capital Markets",
            "containers": "Containers & Packaging",
            "water": "Water Utilities",
            "machinery": "Machinery",
            "beverages": "Beverages",
            "media": "Media" ,
            "interactive": "Interactive Media & Services",
            "automobiles": "Automobiles",
            "hardware": "Technology Hardware, Storage & Peripherals",
            "broadline": "Broadline Retail"
          }

def csv2npz(init_date='2014-01-01',end_date='2024-12-31',
            var_type='close',
            folder='./dat/',
            industry_type='oil'):
    ''' Dado un periodo de tiempos y una industria lee el csv y 
       filtra las compa~nias que tienen datos completos en el periodo
       y los guarda en un npz '''

    sector = sector_d[industry_type]
    if (not os.path.isfile(folder+"stock_metadata.csv") or
        not os.path.isfile(folder+"historical_prices.csv" ) ):
        raise Exception(f"El path: {folder} debe contener los *.csv \n historical_prices.csv") 
    # reading csv file 
    df = pd.read_csv(folder+"stock_metadata.csv")
    df_dat = pd.read_csv(folder+"historical_prices.csv")
    df_company = df[df['industry'] == sector]
    df_dat['date'] = pd.to_datetime(df_dat['date'])

    print('Finished reading csv')
    init_date=pd.to_datetime(init_date)
    end_date = pd.to_datetime(end_date)
    date_range = pd.date_range(start=init_date, end=end_date, freq='D')

    julians, opens, company, volumes = [], [], [], []
    df_all = pd.DataFrame({'date': date_range})

    print('Collecting time series')
    for index, row in df_company.iterrows():
        df_ts = df_dat[df_dat['symbol'] == row['symbol']].copy()

        df_ts=df_ts[(df_ts['date'] >= init_date) & (df_ts['date'] <= end_date)]
        df_ts.loc[:, 'julian'] = (df_ts['date'] - init_date).dt.days

    #    df_ts['julian'] = (df_rangets['date']-init_date).dt.days

        df_rangets=pd.merge(df_all, df_ts[['date', 'julian', var_type,'volume']], on='date', how='left')

        julians.append(df_rangets['julian'].values)
        opens.append(df_rangets[var_type].values)
        volumes.append(df_rangets['volume'].values)
        company.append(row['symbol'])

    day = np.array(julians)
    price = np.array(opens)
    volume = np.array(volumes)
    company = np.array(company)
    print('Cantidad de empresas: ',len(company))
    print(price.shape)
    dt,price,company,volume = clean_data(day,price,company,volume)


    np.savez(folder+f"{industry_type}_{init_date.date()}_{end_date.date()}_day_vol_{var_type}val.npz",
             day=dt,price=price,volume=volume, company=company, startdate=init_date)

if __name__=="__main__":
    csv2npz(init_date='2014-01-01',end_date='2024-12-31')
    
def load_all_ts(sectors=None, pathdat='./dat/',
                init_date='2014-01-01', end_date='2024-12-31'):
    """
    Carga datos de TODOS los sectores (o lista de sectores) y los combina.
    Devuelve day, dates, price, company.
    """
    if sectors is None:
        sectors = list(sector_d.keys())

    all_prices = []
    all_companies = []

    for sec in sectors:
        day, dates, price, company = read_npz(sector=sec, pathdat=pathdat,
                                              init_date=init_date, end_date=end_date)
        all_prices.append(price)
        all_companies.append(company)

    combined_price = np.vstack(all_prices)  # stack all rows
    combined_company = np.concatenate(all_companies)

    # day y dates deberían coincidir en todos los sectores (siempre mismo rango temporal)
    return day, dates, combined_price, combined_company


def load_by_tickers(tickers, pathdat='./dat/'):
    """
    Carga datos para una lista específica de tickers alineados en el máximo rango temporal común.
    Devuelve: day, dates, prices, companies, volumes
    """
    # Cargar metadatos
    metadata_file = os.path.join(pathdat, 'stock_metadata.csv')
    if not os.path.isfile(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    df_meta = pd.read_csv(metadata_file)
    # Filtrar solo los tickers solicitados
    df_tickers = df_meta[df_meta['symbol'].isin(tickers)]
    
    # Verificar tickers faltantes
    missing_tickers = set(tickers) - set(df_tickers['symbol'])
    if missing_tickers:
        print(f"Warning: Tickers not found in metadata: {missing_tickers}")
    
    # Cargar todos los datos históricos
    prices_file = os.path.join(pathdat, 'historical_prices.csv')
    if not os.path.isfile(prices_file):
        raise FileNotFoundError(f"Prices file not found: {prices_file}")
    
    df_prices = pd.read_csv(prices_file, parse_dates=['date'])
    df_prices = df_prices[df_prices['symbol'].isin(tickers)]
    
    # Encontrar fechas comunes
    start_dates = df_prices.groupby('symbol')['date'].min()
    end_dates = df_prices.groupby('symbol')['date'].max()
    
    # Calcular rango común máximo
    common_start = start_dates.max()
    common_end = end_dates.min()
    
    if common_start > common_end:
        raise ValueError("No hay rango temporal común para los tickers seleccionados")
    
    print(f"Rango temporal común: {common_start.date()} a {common_end.date()}")
    
    # Filtrar datos dentro del rango común
    mask = (df_prices['date'] >= common_start) & (df_prices['date'] <= common_end)
    df_common = df_prices[mask]
    
    # Crear matriz de precios y volúmenes
    df_pivot = df_common.pivot(index='date', columns='symbol', values='close')
    df_vol = df_common.pivot(index='date', columns='symbol', values='volume')
    
    # Eliminar activos con datos faltantes
    valid_columns = df_pivot.columns[df_pivot.isna().sum() == 0]
    if len(valid_columns) == 0:
        raise ValueError("No hay activos con datos completos en el rango común")
    
    df_pivot = df_pivot[valid_columns]
    df_vol = df_vol[valid_columns]
    
    # Generar datos de salida
    dates = df_pivot.index.values
    day = (dates - dates[0]).astype('timedelta64[D]').astype(int)
    prices = df_pivot.values
    volumes = df_vol.values
    companies = df_pivot.columns.values
    
    return day, dates, prices.T, companies, volumes.T