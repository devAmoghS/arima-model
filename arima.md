__What does ARIMA(1, 0, 12) mean?__

Specifically for your model, __ARIMA(1, 0, 12)__ means that it you are 
describing some response variable (Y) by combining a 1st order Auto-Regressive model 
and a 12th order Moving Average model. A good way to think about it is (AR, I, MA). 
This makes your model look the following, in simple terms:

__Y = (Auto-Regressive Parameters) + (Moving Average Parameters)__

The 0 in the between the 1 and the 12 represents the 'I' part of the model (the Integrative part) 
and it signifies a model where you're taking the difference between response variable data - 
this can be done with non-stationary data and it doesn't seem like you're dealing with that, 
so you can just ignore it.


__What values can be assigned to p, d, q?__

Lots of different whole numbers. There are diagnostic tests you can do to try to find the best values of p,d,q (see part 3).

__What is the process to find the values of p, d, q?__

There are a number of ways, and I don't intend this to be exhaustive:

look at an autocorrelation graph of the data (will help if Moving Average (MA) model is appropriate)
look at a partial autocorrelation graph of the data (will help if AutoRegressive (AR) model is appropriate)
look at extended autocorrelation chart of the data (will help if a combination of AR and MA are needed)
try Akaike's Information Criterion (AIC) on a set of models and investigate the models with the lowest AIC values
try the Schwartz Bayesian Information Criterion (BIC) and investigate the models with the lowest BIC values
