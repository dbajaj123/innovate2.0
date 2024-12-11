model = Sequential()
model.add(Dense(100, activation='sigmoid',input_dim=14))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
