from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from utils.loadData import dataset
from utils.menu import plot_options

def forestRegressor():
    png_name = 'forestRegressor'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria']
    target_cols = ['gravidade']
    
    ## Funciona pra 50 arvores
    n_estimators = 25

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=n_estimators, criterion='poisson', max_depth=20)

    reg = reg.fit(x_train, y_train)

    predict = reg.predict(x_test)
    print(predict)
    
    score = r2_score(y_test, predict)
    print(score)    
    
    score = r2_score(y_test, predict)
    print("r2 score", score)
    
    mse = mean_squared_error(y_test, predict)
    print("MSE:", mse)

    plot_options(reg, feature_cols, png_name,  type='forest', classifier='false', n_estimators=n_estimators)

