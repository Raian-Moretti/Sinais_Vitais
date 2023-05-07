from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
from utils.loadData import dataset
from utils.menu import plot_options

def treeRegressor():
    png_name = 'treeRegressor'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria']
    target_cols = ['gravidade']

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=42)

    reg = DecisionTreeRegressor(criterion='poisson', max_depth=20)

    reg = reg.fit(x_train, y_train)

    predict = reg.predict(x_test)
    print(predict)
    
    score = r2_score(y_test, predict)
    print("r2 score", score)
    
    mse = mean_squared_error(y_test, predict)
    print("MSE:", mse)
    
    plot_options(reg, feature_cols, png_name, type='tree')
