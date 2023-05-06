from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from utils.loadData import dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.plot import confusion_matrix

def neuralClassifier():
    png_name = 'neuralClassifier'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade']
    target_cols = ['estado']

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=42)
    
    clf = MLPClassifier(hidden_layer_sizes=(4,8,16,32,16,8,4), activation="relu", max_iter=1500, learning_rate='adaptive', learning_rate_init=0.001)

    clf = clf.fit(x_train, y_train.values.ravel())
    
    predict = clf.predict(x_test)
    print(predict)
    
    accuracy = accuracy_score(y_test, predict)
    print("accuracy:", accuracy)
    
    precision = precision_score(y_test, predict, average=None)
    print("precision:", precision)
    
    recall = recall_score(y_test, predict, average=None)
    print("recall:", recall)
    
    f1 = f1_score(y_test, predict, average=None)
    print("f1 score:", f1)

    confusion_matrix(clf, x_test, y_test,png_name)

