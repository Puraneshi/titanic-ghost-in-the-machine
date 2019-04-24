import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

df = pd.read_csv('TabelaNumericos.csv')
y = df.pop('Survived')

testes = []
ini, fim, passo = 300, 500, 50
for i in range(ini, fim, passo):
    print((i-ini)/(fim-ini))
    model = RandomForestRegressor(i,
                                  oob_score=True,
                                  n_jobs=-1,
                                  random_state=42,
                                  min_samples_leaf=6)
    model.fit(df, y)
    roc = roc_auc_score(y, model.oob_prediction_)
    testes.append((i, roc))

final = pd.DataFrame(testes, columns=['iter', 'roc'])
print(final.sort_values('roc', ascending=False))
plt.plot(final.iter, final.roc)
plt.show()
