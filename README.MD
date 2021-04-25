# Глубокое обучение в анализе рынка
Это обзорный перевод. [оригинал](https://medium.com/swlh/predicting-stock-prices-using-deep-learning-models-310b41cec90a)

Основной упор в статье делается на рекуррентные нейронные сети и библиотеку `keras`, как предоставленную реалиацию.
В статье используются данные о ценах некоторых акций с 2006 по 2018 год. Целью же исследования поставлено прогнозирование цен за 2017 год на осннове данных за 2006 - 2016 год. 
Полный датасет доступен по [ссылке](https://www.kaggle.com/szrlee/stock-time-series-20050101-to-20171231).

![Image of Yaktocat](https://miro.medium.com/max/3000/1*KlDWtInyDwHkFk_xP5j8VA.png)

В статье выбрана наивысшая цена за день в качестве целевой переменной. 

![результаты](https://miro.medium.com/max/3000/1*VLKREecu-oOihUqzfJ1ONQ.png)

В ходе исследования авторы статьи пришли к следующим результатам.

Следует отметить особенность анализа временных рядов, к которым относятся, в том числе, и акции. 
Значение зависит не от параметров, а от предыдущих значений, поэтому важно использовать в анализе тренды.
Именно из-за зависимости от предыдущих значений, рекуррентные нейронные сети и LSTM, являются подходящими инструментами для анализа цен акций.

В библиотеке keras есть несколько методов, реализующие рекуррентность: RNN, LSTM, GRU.


## Предобработка данных

Прежде всего необходимо разбить выборку на обучающую и тестовую

```python
training_set = all_data[:'2016'].iloc[:,1:2].values
test_set = all_data['2017':].iloc[:,1:2].values
```
Затем проскалируем данные
```python
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
```

В статье используется промежуток в 60 дней для определения цены.
Полный код подготовки данных
```python
def create_dl_train_test_split(all_data):
    '''
    input: 
      all_data: dataframe with dates and price data
     
    output:
      X_train: data from 2006-2017
      y_train: data from 2017-2018
      X_test:  data from 2017-2018
      sc:      insantiated MinMaxScaler object fit to the training data
    '''
    # create training and test set
    training_set = all_data[:'2016'].iloc[:,1:2].values
    test_set = all_data['2017':].iloc[:,1:2].values

    # scale the data
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)

    # create training and test data
    X_train = []
    y_train = []
    for i in range(60,2768):
        X_train.append(training_set_scaled[i-60:i,0])
        y_train.append(training_set_scaled[i,0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping X_train for efficient modelling
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))

    total_data = pd.concat((all_data["High"][:'2016'], all_data["High"]['2017':]),axis=0)
    inputs = total_data[len(total_data)-len(test_set) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = sc.transform(inputs)

    # Preparing X_test
    X_test = []
    for i in range(60,311):
        X_test.append(inputs[i-60:i,0])
        
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    return X_train, y_train, X_test, sc
``` 
## RNN, LSTM, GRU

Сейчас попробуем разные способы которые предоставляет keras. Начнем с более простой модели с 1 слоем.

```python

def create_single_layer_small_rnn_model(X_train, y_train, X_test, sc):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    model = Sequential()
    model.add(SimpleRNN(6))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    model.fit(X_train, y_train, epochs=100, batch_size=150)

    # Finalizing predictions
    scaled_preds = model.predict(X_test)
    test_preds = sc.inverse_transform(scaled_preds)

    return model, test_preds
```

Далее добавим нейронов, но оставим прежнее количество слоев.

```python
def create_single_layer_rnn_model(X_train, y_train, X_test, sc):
    '''
    create single layer rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    model = Sequential()
    model.add(SimpleRNN(32))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    model.fit(X_train, y_train, epochs=100, batch_size=150)

    # Finalizing predictions
    scaled_preds = model.predict(X_test)
    test_preds = sc.inverse_transform(scaled_preds)

    return model, test_preds
```

Добавим слоев

```python
def create_rnn_model(X_train, y_train, X_test, sc):
    '''
    create rnn model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # create a model
    model = Sequential()
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32, return_sequences=True))
    model.add(SimpleRNN(32))
    model.add(Dense(1))

    model.compile(optimizer='rmsprop', loss='mean_squared_error')

    # fit the RNN model
    model.fit(X_train, y_train, epochs=100, batch_size=150)

    # Finalizing predictions
    scaled_preds = model.predict(X_test)
    test_preds = sc.inverse_transform(scaled_preds)

    return model, test_preds
```

Модели с малым количеством слоев могут быть недообучены, а с большим переобучены.
Следует отметить, что при создании многослойных моделей нужно указывать `return_sequences=True` с первого по предпоследний слой, так как SimpleRNN может быть наиболее полезен, когда самое последее наблюдение хранит необходимую информацию для предсказания результата.
Модели, использующие GRU или LSTM могут превзойти модели использующие SimpleRNN, потому что эти модели могут более долгосрочные зависимости.
Их преимущества достигаются засчет решения проблемы исчезающего градиента.

На практике LSTM и GRU дают примерно одинаковый результат. Небольшое преимущество получает GRU засчет более простых вычислений.

Ниже пример кода с GRU.
```
def create_GRU_model(X_train, y_train, X_test, sc):
    '''
    create GRU model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(GRU(units=50, activation='tanh'))
    regressorGRU.add(Dense(units=1))

    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    # Fitting to the training set
    regressorGRU.fit(X_train,y_train,epochs=50,batch_size=150)

    GRU_predicted_stock_price = regressorGRU.predict(X_test)
    GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)

    return regressorGRU, GRU_predicted_stock_price
```

Сложные нейронные сети легко переобучаются. Для решения этой проблемы можно использовать дропауты.
В нашем случае рекомендуется применять дропауты к dense-слоям, которые следуют за рекуррентными.
Дропаут не рекомендуется применять к рекуррентным слоям, так как возможны потери данных.

Ниже пример применения дропаутов

```python
def create_GRU_with_drop_out_model(X_train, y_train, X_test, sc):
    '''
    create GRU model trained on X_train and y_train
    and make predictions on the X_test data
    '''
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(GRU(units=50, return_sequences=True, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    
    # Third GRU layer
    regressorGRU.add(GRU(units=50, return_sequences=True, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(GRU(units=50, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(units=1))
    # Compiling the RNN
    regressorGRU.compile(optimizer=SGD(lr=0.01, decay=1e-7, momentum=0.9, nesterov=False),loss='mean_squared_error')
    # Fitting to the training set
    regressorGRU.fit(X_train,y_train,epochs=50,batch_size=150)

    GRU_predicted_stock_price = regressorGRU.predict(X_test)
    GRU_predicted_stock_price = sc.inverse_transform(GRU_predicted_stock_price)

    return regressorGRU, GRU_predicted_stock_price
```

Так же автор статьи использовал Facebook prophet, который учитывает такие параметры, как человеческий фактор, исторические тренды.

```python
def create_prophet_results(all_data,
                           final_train_idx=2768,
                           pred_periods=250):
    '''
    create prophet model trained on first 2768 rows by
    default and predicts on last 250 rows
    '''
    # Pull train data
    train_data = all_data[:final_train_idx].reset_index()[['Date', 'High']]
    train_data.columns = ['ds', 'y']

    # Create and fit model
    prophet_model = Prophet()
    prophet_model.fit(train_data)

    # Provide predictions
    test_dates = prophet_model.make_future_dataframe(periods=pred_periods)
    forecast_prices = prophet_model.predict(test_dates)

    return forecast_prices
```

В итоге использовав все модели автор пришел к следующему результату

![](https://miro.medium.com/max/3000/1*KcVERlBL76xl0ZbDNazqdQ.png)

Как можно убедится рекуррентные нейронные сети хорошо справляются с задачей предсказания цен акций.