import pandas as pd

arquivo = input('digite nome do csv:\n')
df = pd.read_csv(arquivo)

df['Age'].fillna(df.Age.mean(), inplace=True)
df.drop(['PassengerId'], axis=1, inplace=True)


def cabineLetra(element):
    try:
        return element[0]
    except TypeError:
        return "None"


df['Cabin'] = df.Cabin.apply(cabineLetra)
trocaVal = ['Sex', 'Embarked', 'Cabin']
for v in trocaVal:
    df[v].fillna("Missing", inplace=True)
    df = pd.concat([df, pd.get_dummies(df[v], prefix=v)], axis=1)
    df.drop([v], axis=1, inplace=True)

num_val = list(df.dtypes[df.dtypes != 'object'].index)

df[num_val].to_csv('numericos'+arquivo, index=False)
