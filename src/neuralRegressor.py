from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from utils.loadData import dataset

def neuralRegressor():
    png_name = 'neuralRegressor'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade']
    target_cols = ['estado']

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=1)
    
    reg = MLPRegressor(hidden_layer_sizes=(32,128,128,64,32,8),activation="relu" ,random_state=1, max_iter=500)

    reg = reg.fit(x_train, y_train.values.ravel())
    
    predict = reg.predict(x_test)

    score = r2_score(predict, y_test)

    print(score)
