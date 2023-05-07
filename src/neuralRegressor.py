from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from utils.loadData import dataset
from utils.menu import plot_options

def neuralRegressor():
    png_name = 'neuralRegressor'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria']
    target_cols = ['gravidade']

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
 
    reg = MLPRegressor(hidden_layer_sizes=(32,64,128,64,32,8), activation="relu", max_iter=1500, learning_rate='adaptive', learning_rate_init=0.001)

    reg = reg.fit(x_train, y_train.values.ravel())
    
    predict = reg.predict(x_test)
    print(predict)
    
    score = r2_score(y_test, predict)
    print("r2 score", score)
    
    mse = mean_squared_error(y_test, predict)
    print("MSE:", mse)
    
    plot_options(reg, feature_cols, png_name, x_test, y_test, type='neural')


