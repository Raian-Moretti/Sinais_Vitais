from utils.plot import plot, importance

def plot_options(clf,feature_cols, png_name, n_estimators=1):
    opt = -1
    while opt != 0:
        print("\nO que deseja?\n1 - Gerar imagem da estrutura\n2 - Gráfico de importância de feature\n0 - Sair")
        try:
            opt = int(input())
        except ValueError:
            print("\nFavor digitar um número")
            continue
        if(opt == 1):
            plot(clf,feature_cols, png_name, n_estimators)
        elif(opt == 2):
            importance(clf)
        elif(opt == 0):
            print("\nFinalizado!")
        else:
            print("\nOpção inválida")