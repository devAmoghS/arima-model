import warnings

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA


def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


sub_series = pd.read_csv('dataset_abli.csv',
                         header=0,
                         squeeze=True
                         )

month_int = {
    'JAN': 1,
    'FEB': 2,
    'MAR': 3,
    'APR': 4,
    'MAY': 5,
    'JUN': 6,
    'JUL': 7,
    'AUG': 8,
    'SEP': 9,
    'OCT': 10,
    'NOV': 11,
    'DEC': 12,

}

sub_series.MONTH = sub_series.MONTH.map(month_int)
sub_series['Date'] = pd.to_datetime(sub_series.YEAR.astype(str) + '-' + sub_series.MONTH.astype(str))
sub_series = sub_series.drop(['MONTH', 'YEAR'], axis=1)

print(sub_series.info())

sub_series.set_index('Date', inplace=True)

# evaluate parameters
# p_values = [0, 1, 2, 4, 6, 8, 10]
# d_values = range(0, 3)
# q_values = range(0, 3)
# warnings.filterwarnings("ignore")
# evaluate_models(sub_series.values, p_values, d_values, q_values)

X = sub_series.values
size = int(len(X) * 0.66)
# splitting as 66: 33 ratio
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

for t in range(len(test)):
    # order contains the parameter (p,d,q)
    model = ARIMA(history, order=(4, 2, 1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()

    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plotting the data
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
