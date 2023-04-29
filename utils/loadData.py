import pandas as pd

dataset = []

col = ['id','pressao_sistolica','pressao_diastolica','qualidade_da_pressao','pulso','frequencia_respiratoria','gravidade','estado']

dataset = pd.read_csv('data/treino_sinais_vitais_com_label.txt', header=None, names=col)