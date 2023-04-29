from src.treeRegressor import treeRegressor
from src.treeClassifier import treeClassifier
from src.forestRegressor import forestRegressor
from src.forestClassifier import forestClassifier

opt = -1
while opt != 0:
    print("\nQual algoritmo deseja executar?\n1 - Tree Regressor\n2 - Tree Classifier\n3 - Forest Regressor\n4 - ForestClassifier\n0 - Sair")
    try:
        opt = int(input())
    except ValueError:
        print("\nFavor digitar um número")
        continue
    if(opt == 1):
        treeRegressor()
    elif(opt == 2):
        treeClassifier()
    elif(opt == 3):
        forestRegressor()
    elif(opt == 4):
        forestClassifier()    
    elif(opt == 0):
        print("\nFinalizado!")
    else:
        print("\nOpção inválida")