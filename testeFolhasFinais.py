import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

df = pd.read_csv('TabelaNumericos.csv')
y = df.pop('Survived')

testes = {}
ini, fim, passo = 1, 10, 1
for i in range(ini, fim, passo):
    print(i/fim)
    model = RandomForestRegressor(n_estimators=1000,
                                  oob_score=True,
                                  n_jobs=-1,
                                  random_state=42,
                                  min_samples_leaf=i)
    model.fit(df, y)
    roc = roc_auc_score(y, model.oob_prediction_)
    testes[i] = roc

final = pd.Series(testes)
print(final)
final.plot()
plt.show()