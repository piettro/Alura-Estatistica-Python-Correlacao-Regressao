import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import t as t_student

dataset = {
    'Y': [3011, 1305, 1879, 2654, 2849, 1068, 2892, 2543, 3074, 849, 2184, 2943, 1357, 2755, 2163, 3099, 1600, 353, 1778, 740, 2129, 3302, 2412, 2683, 2515, 2395, 2292, 1000, 600, 1864, 3027, 1978, 2791, 1982, 900, 1964, 1247, 3067, 700, 1500, 3110, 2644, 1378, 2601, 501, 1292, 2125, 1431, 2260, 1770],
    'X': [9714, 3728, 6062, 8845, 8378, 3338, 8507, 7947, 9915, 1632, 6825, 8918, 4100, 9184, 6180, 9997, 4500, 1069, 5925, 2466, 6083, 9712, 7780, 8383, 7185, 7483, 7640, 2100, 2000, 6012, 8902, 5345, 8210, 5662, 2700, 6546, 2900, 9894, 1500, 5000, 8885, 8813, 3446, 7881, 1164, 3401, 6641, 3329, 6648, 4800]
}

dataset = pd.DataFrame(dataset)

Y = dataset.Y
X = sm.add_constant(dataset.X)

model  = sm.OLS(Y,X, missing='drop')
result = model.fit()

dataset['Y_predict'] = result.predict()
dataset['u'] = result.resid

print(dataset.head())
print(dataset.u.mean())

##OLS with only one exog var
ax = sns.scatterplot(x=dataset.X, y=dataset.u)
ax.figure.set_size_inches(12, 6)
ax.set_title('u vs Exog', fontsize=18)
ax.set_xlabel('X', fontsize=14)
ax.set_ylabel('u', fontsize=14)
plt.show()

##OLS with more than one exog var
ax = sns.scatterplot(x=dataset.Y_predict, y=dataset.u)
ax.figure.set_size_inches(12, 6)
ax.set_title('u vs Y Predict', fontsize=18)
ax.set_xlabel('Y Predict', fontsize=14)
ax.set_ylabel('u', fontsize=14)
plt.show()

sqe = result.ssr
sqr = result.ess
r2 = result.rsquared
R2Adjusted = result.rsquared_adj

print(f'SQE: {sqe}, SQR: {sqr}, R2: {r2}, R2Adjusted: {R2Adjusted}')

print(result.summary())

eqm = result.mse_resid
print(f'EQM {eqm}')

##Test sig
s = np.sqrt(result.mse_resid)
sum_desv2 = dataset.X.apply(lambda x: (x - dataset.X.mean())**2).sum()
s_beta_2 = s / np.sqrt(sum_desv2)

confidenc = 0.95
significanc = 1 - confidenc
degree_freedom = result.df_resid
probability = (0.5 + (confidenc / 2))

t_alpha_2 = t_student.ppf(probability, degree_freedom)
t = (result.params.iloc[1] - 0) / s_beta_2

print(t <= -t_alpha_2)
print(t >= t_alpha_2)

p_value = 2 * (t_student.sf(t, degree_freedom))
print(f'pvalue: {p_value}')

p_value = result.pvalues.iloc[1]
print(f'pvalue: {p_value}')

F = result.fvalue
F_p_value = result.f_pvalue

print(f'SQR/k: {result.mse_model}')
print(f'SQE / (n-k-1): {result.mse_resid}')
print(f'F: {F}')
print(f'Reject F p value: {F_p_value <= 0.05}')