from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils.loadData import dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from utils.menu import plot_options

def forestClassifier():
    png_name = 'forestClassifier'
    feature_cols = ['qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade']
    target_cols = ['estado']

    ## Funciona pra 50 arvores
    n_estimators = 25

    features = dataset[feature_cols]
    target = dataset[target_cols]

    x_train, x_test, y_train, y_test = train_test_split(features, target,  test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=n_estimators, criterion='gini', max_depth=3)

    clf = clf.fit(x_train, y_train.values.ravel())

    predict = clf.predict(x_test)
    print(predict)
    
    accuracy = accuracy_score(y_test, predict)
    print("accuracy", accuracy)
    
    precision = precision_score(y_test, predict, average=None)
    print("precision:", precision)
    
    recall = recall_score(y_test, predict, average=None)
    print("recall:", recall)
    
    f1 = f1_score(y_test, predict, average=None)
    print("f1 score:", f1)

    plot_options(clf, feature_cols, png_name, n_estimators)

