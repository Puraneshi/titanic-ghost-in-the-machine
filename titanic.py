import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

# le o dataset e remove a coluna objetivo
df = pd.read_csv('train.csv')
y = df.pop("Survived")

# preenche linhas de idades vazias com a media
df['Age'].fillna(df.Age.mean(), inplace=True)
df.drop(['PassengerId'], axis=1, inplace=True)
# print(df.Age.head(10))
# print(df.describe())

def cabineLetra(element):
    # retorna primeira letra da string ou 'None'
    try:
        return element[0]
    except TypeError:
        return "None"


df['Cabin'] = df.Cabin.apply(cabineLetra)

# transforma 'sex', 'embarked' e 'cabin', strings, em colunas de 0 ou 1
trocaVal = ['Sex', 'Embarked', 'Cabin']
for v in trocaVal:
    df[v].fillna("Missing", inplace=True)
    df = pd.concat([df, pd.get_dummies(df[v], prefix=v)], axis=1)
    df.drop([v], axis=1, inplace=True)

# lista das colunas que possuem valores numericos
num_val = list(df.dtypes[df.dtypes != 'object'].index)
'''
# faz o mesmo que o codigo acima
num_val = []
for t in df:
    if df[t].dtype != 'object':
        num_val.append(t)
'''

# comeca modelo inicial
model = RandomForestRegressor(n_estimators=1000,
                              oob_score=True,
                              n_jobs=-1,
                              min_samples_leaf=5)
# treina modelo
model.fit(df[num_val], y)
print(model.oob_score_)

# verifica previsões 'out of bag'
y_oob = model.oob_prediction_
print("roc: {}".format(roc_auc_score(y, y_oob)))

# organiza e imprime colunas e taxa de importancia
features = pd.Series(model.feature_importances_, index=df[num_val].columns)

# une as variaveis do mesmo tipo(que anteriormente eram strings)
repetidas = {}
for name in trocaVal:
    repetidas[name] = []
    for col in features.keys():
        if name in col:
            repetidas[name].append(col)
print(repetidas)
for item in repetidas:
    features[item] = sum(features[repetidas[item]])
    features.drop(repetidas[item], inplace=True)

# ordena decrescente
features = features.sort_values(ascending=False)
print(features)

# plot das caracteristicas importantes
ax = features.plot(kind='barh')
ax.invert_yaxis()
plt.tight_layout()
plt.show()
