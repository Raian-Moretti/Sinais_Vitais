from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from utils.loadData import dataset
from utils.plot import confusion_matrix

def neuralClassifier():
    png_name = 'neuralClassifier'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade']
    target_cols = ['estado']

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=1)
    
    reg = MLPClassifier(hidden_layer_sizes=(4,8,16,8,4),activation="relu" ,random_state=1, max_iter=500)

    reg = reg.fit(x_train, y_train.values.ravel())
    
    predict = reg.predict(x_test)

    confusion_matrix(reg, x_test, y_test,png_name)

