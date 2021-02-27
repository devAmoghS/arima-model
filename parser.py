import pandas as pd
import matplotlib.pyplot as plt


def parser(x):
    return pd.datetime.strptime('190' + x, '%Y-%m')


series = pd.read_csv('shampoo_sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
series.plot()
plt.show()
