# -*- coding: utf-8 -*-
"""
Amin Darbandi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={"figure.figsize": (16, 8), "figure.dpi": 300})
sns.set_style("dark", {'axes.grid' : False})
plt.rcParams['axes.facecolor'] = '#e8e6e6'


df = pd.read_csv('Metrics.csv', delimiter='\t', index_col=0, 
                 usecols=[0,8,9,10,11,12,13,14,15,16])
df = df.reset_index()

x = np.arange(len(df))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(layout='constrained')
ax.bar(x=x , height = df['rmee val'], width=0.2, label = 'RMEE', 
       color = '#0e87cc')
ax.bar(x=x+0.11 , height = df['mae val'], width=0.2, label = 'MAE',
       color = '#c14a09')
ax.bar(x=x+0.22 , height = df['mse val'], width=0.2, label = 'MSE', 
       color = '#154406')
twin_ax = ax.twinx()
twin_ax.plot(df['R2 val']*100, label= 'R2', color ='#4e0550')
twin_ax.set_ylabel('R2 [%]', fontsize=14)
ax.set_xticks(x, df.index, rotation=0, fontsize=14)
# ax.yaxis.set_ticks_position('none')

ax.set_title('Root Mean Squared, Mean Absolute and Mean  \n'
          'Squared Error of validation set', fontsize=16)
# ax.text(x=6, y=-0.04, s='LSTM Cases', fontsize=16)
ax.set_xlabel('LSTM Case', fontsize=16)

handels, labels = ax.get_legend_handles_labels()
handels_1, labels2 = twin_ax.get_legend_handles_labels()
ax.legend(handels + handels_1, labels + labels2, 
          bbox_to_anchor=(0.5, 0.3, 0.5, 0.5),
          frameon=False,
          fontsize=14)

# ax.grid(False)
# twin_ax.grid(False)



fig, ax = plt.subplots(layout='constrained')
ax.bar(x=x , height = df['rmee test with 1 batch'], width=0.2, 
       label = 'RMEE', color = '#0e87cc')
ax.bar(x=x+0.11 , height = df['mae test with 1 batch'], width=0.2, 
       label = 'MAE', color = '#c14a09')
ax.bar(x=x+0.22 , height = df['mse test with 1 batch'], width=0.2, 
       label = 'MSE', color = '#154406')
twin_ax = ax.twinx()
twin_ax.plot(df['R2 test']*100, label= 'R2', color ='#4e0550')
twin_ax.set_ylabel('R2 [%]', fontsize=14)
ax.set_xticks(x, df.index, rotation=0, fontsize=14)
# ax.yaxis.set_ticks_position('none')

ax.set_title('Root Mean Squared, Mean Absolute and Mean  \n'
          'Squared Error of test set', fontsize=16)
# ax.text(x=6, y=-0.04, s='LSTM Cases', fontsize=16)

handels, labels = ax.get_legend_handles_labels()
handels_1, labels2 = twin_ax.get_legend_handles_labels()
ax.legend(handels + handels_1, labels + labels2, 
          bbox_to_anchor=(0.5, 0.3, 0.5, 0.5), 
          frameon=False,
          fontsize=14)
ax.set_xlabel('LSTM Case', fontsize=16)
# ax.grid(False)
# twin_ax.grid(False)


fig, ax = plt.subplots(layout='constrained')
ax.bar(x=x[6:] , height = df['MAPE test with 1 batch'].iloc[6:], width=0.2, 
       label = 'MAPE', color = '#0e87cc')
ax.set_xticks(x[6:], df.Name.iloc[6:], rotation=90, fontsize=14)

ax.set_title('Mean Absolute Percentage Error of test set', fontsize=16)
ax.text(x=10.5, y=-12, s='LSTM Cases', fontsize=16)
ax.set_ylabel('MAPE [ %]' , fontsize=14)
ax.set_yticklabels(ax.get_yticks(), fontsize=14)
handels, labels = ax.get_legend_handles_labels()
ax.legend(handels, labels, 
          loc=0, 
          frameon=False,
          fontsize=14)
