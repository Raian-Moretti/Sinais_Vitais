from sklearn.ensemble import RandomForestClassifier
import csv
import matplotlib.pyplot as plt

labeled_data = None

# Carrega os dados com label
with open("treino_sinais_vitais_com_label.txt", "r") as f:
    labeled_data = [list(map(float, row)) for row in csv.reader(f, delimiter=",")]
labeled_features = [row[1:-1] for row in labeled_data]
labeled_target = [row[-1] for row in labeled_data]

# Carrega os dados sem label
with open("teste_sinais_vitais_sem_label.txt", "r") as f:
    unlabeled_data = [list(map(float, row)) for row in csv.reader(f, delimiter=",")]
    data = [row for row in unlabeled_data]
    unlabeled_data = [row[1:-1] for row in data]
    validate = [row[-1] for row in data]

# Cria a floresta aleatória
clf = RandomForestClassifier(n_estimators=100, criterion='gini')

# Treina a floresta aleatória
clf.fit(labeled_features, labeled_target)

# Faz a predição dos dados sem label
predicted_target = clf.predict(unlabeled_data)

# Imprime os resultados
print(predicted_target[:10])

# Verifica se a predição bate com as labels existentes
for i in range(len(predicted_target)):
    print(predicted_target[i]-validate[i])