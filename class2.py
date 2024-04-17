import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv('data/dados.csv')
print(data.head())

sample = data.query('Renda < 5000').sample(n=20, random_state=101)
cov_sample = sample[['Idade','Renda','Anos de Estudo','Altura']].cov()

print(cov_sample)

#================================================
x = sample.Renda
y = sample.Idade

ax = sns.scatterplot(x, y)
ax.figure.set_size_inches(10, 6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors='black', linestyles='dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors='black', linestyles='dashed')
plt.show()

#================================================
x = sample.Renda
y = sample['Anos de Estudo']

ax = sns.scatterplot(x, y)
ax.figure.set_size_inches(10, 6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors='black', linestyles='dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors='black', linestyles='dashed')
plt.show()

#================================================
x = sample.Idade
y = sample.Altura

ax = sns.scatterplot(x, y)
ax.figure.set_size_inches(10, 6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors='black', linestyles='dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors='black', linestyles='dashed')
plt.show()

#================================================
s_xy = data[['Altura', 'Renda']].cov()
print(s_xy)

s_x = data.Altura.std()
s_y = data.Renda.std()

r_xy = s_xy / ( s_x * s_y)
print(r_xy)

corr_altura_renda = data[['Altura', 'Renda']].corr()
print(corr_altura_renda)

x = sample.Renda
y = sample.Altura

ax = sns.scatterplot(x, y)
ax.figure.set_size_inches(10, 6)
ax.hlines(y = y.mean(), xmin = x.min(), xmax = x.max(), colors='black', linestyles='dashed')
ax.vlines(x = x.mean(), ymin = y.min(), ymax = y.max(), colors='black', linestyles='dashed')
plt.show()
