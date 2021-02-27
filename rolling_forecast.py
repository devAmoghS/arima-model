import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


def parser(x):
    return pd.datetime.strptime('190' + x, '%Y-%m')


series = pd.read_csv('shampoo_sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

X = series.values
# train-test split => 66::33
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]

history = [x for x in train]
predictions = list()

# fit model
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()

    y_hat = output[0]
    predictions.append(y_hat)
    obs = test[t]
    history.append(obs)

    print('predicted=%f, expected=%f' % (y_hat, obs))

# compute MSE
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot predictions
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
