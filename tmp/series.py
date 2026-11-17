import os,io,zipfile,requests,pandas as pd,yfinance as yf
os.makedirs('tmp/out',exist_ok=True)
H={'User-Agent':'Mozilla/5.0'}
k=os.environ.get('FRED_API_KEY')
for s in ['DGS10','DFII10','T10YIE','CPIAUCSL','BAA10Y']:
    try:
        r=requests.get('https://api.stlouisfed.org/fred/series/observations',params={'series_id':s,'api_key':k,'file_type':'json','frequency':'m','aggregation_method':'avg'},timeout=60)
        d=pd.DataFrame(r.json()['observations'])[['date','value']];d.to_csv(f'tmp/out/fred_{s}.csv',index=False);print(s,len(d))
    except Exception as e: print('fred',s,e)
for u,n in [('http://www.econ.yale.edu/~shiller/data/ie_data.xls','ie_data.xls'),('https://pages.stern.nyu.edu/~adamodar/pc/datasets/histimpl.xls','histimpl.xls'),('https://www.stern.nyu.edu/~adamodar/pc/datasets/histimpl.xls','histimpl2.xls')]:
    try:
        r=requests.get(u,headers=H,timeout=60);print(n,r.status_code,len(r.content))
        if r.ok: open('tmp/out/'+n,'wb').write(r.content)
    except Exception as e: print(n,e)
base='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
for z in ['Portfolios_Formed_on_BE-ME_CSV.zip','F-F_Research_Data_Factors_CSV.zip','Portfolios_Formed_on_E-P_CSV.zip','6_Portfolios_2x3_CSV.zip']:
    try:
        r=requests.get(base+z,headers=H,timeout=120);print(z,r.status_code,len(r.content))
        zf=zipfile.ZipFile(io.BytesIO(r.content))
        for nm in zf.namelist(): open('tmp/out/'+nm,'wb').write(zf.read(nm))
    except Exception as e: print(z,e)
T=['SPY','ACWI','GOOGL','MSFT','META','AMZN','CRM','ADBE','JPM','MA','V','NU','EWZ','EEM','TSM','ANET','AVGO','PEP','MCD','TMUS','BRK-B','XOM','CVX','NEM','B','GOLD','WMT','COST','KO','PG','PM','JNJ','MU','IWF','IWD','XLK','XLP','XLF','XLE','XLV','XLU','XLI','XLY','GLD','IEF','ORCL','IBM','CSCO','INTC','AAPL','NVDA','QCOM','TXN']
d=yf.download(T,start='1995-01-01',interval='1mo',auto_adjust=True,progress=False)['Close']
d.to_csv('tmp/out/mensual_tickers.csv');print('yf',d.shape)
