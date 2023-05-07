from utils.plot import plot, importance, confusion_matrix, loss

def plot_options(alg, feature_cols, png_name, x_test=None, y_test=None, type='', classifier='false', n_estimators=1):
    opt = -1
    while opt != 0:
        print(f"\nO que deseja?\n1 - Gerar imagem da estrutura\n2 - Gráfico de importância de feature\n3 - Matriz de Confusão\n4 - Loss\n0 - Voltar")
        try:
            opt = int(input())
        except ValueError:
            print("\nFavor digitar um número")
            continue
        if(opt == 1 and (type=='tree' or type=='forest') and classifier=='true'):
            plot(alg,feature_cols, png_name, n_estimators)
        elif(opt == 2 and (type=='tree' or type=='forest')):
            importance(alg,png_name)
        elif(opt == 3 and classifier=='true'):
            confusion_matrix(alg, x_test, y_test, png_name)     
        elif(opt == 4 and (type=='neural')):
            loss(alg, png_name)  
        elif(opt == 0):
            print("\n### Menu inicial ###")
        else:
            print("\nOpção inválida")