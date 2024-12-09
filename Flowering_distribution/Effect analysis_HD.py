import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.font_manager as fm 
import os

mpl.rcParams['font.family'] = 'Times New Roman'


current_dir = os.path.dirname(os.path.realpath(__file__))

file_path = os.path.join(current_dir, 'Fertility-Temperature.xlsx')
df = pd.read_excel(file_path)

data_summary = df.head()

grouped_by_year = df.groupby('Year').size()

model_one_way = ols('HD ~ C(Year)', data=df).fit()
anova_results_one_way = anova_lm(model_one_way, typ=2)

X = df[['Tma', 'Tna', 'Tda']]  
y = df['HD']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)  
X_pca = pca.fit_transform(X_scaled)

pca_components = pca.components_

X_pca = sm.add_constant(X_pca)

model_pca = sm.OLS(y, X_pca).fit()

pca_regression_summary = model_pca.summary()

output_directory = current_dir

data_summary.to_excel(f'{output_directory}\\data_summary.xlsx', index=False)

grouped_by_year.to_csv(f'{output_directory}\\grouped_by_year.csv', header=True)

anova_results_one_way.to_csv(f'{output_directory}\\anova_results_one_way.csv')

pca_components_df = pd.DataFrame(pca_components, columns=['PC1', 'PC2', 'PC3'], index=['Tma', 'Tna', 'Tda'])
pca_components_df.to_csv(f'{output_directory}\\pca_components.csv')

with open(f'{output_directory}\\pca_regression_summary.txt', 'w') as f:
    f.write(str(pca_regression_summary))

plt.figure(figsize=(10, 10))
sns.violinplot(
    x='Year', 
    y='HD', 
    data=df, 
    inner="quartile",
    palette=[(0/255, 128/255, 128/255)] * len(df['Year'].unique()),
    alpha=0.6
)

plt.xlabel('Year', fontsize=32, color='black', weight='bold', labelpad=25)
plt.ylabel('Heading duration (d)', fontsize=32, color='black', weight='bold', labelpad=25)

plt.xticks(fontsize=24, color='black', weight='bold')
plt.yticks(fontsize=24, color='black', weight='bold')

ax = plt.gca() 
ax.set_facecolor('none') 
for spine in ax.spines.values():
    spine.set_edgecolor('black') 
    spine.set_linewidth(3)      
    spine.set_linestyle('-') 
plt.tick_params(axis='x', 
                direction='out', 
                length=10,    
                width=3,          
                colors='black')    
plt.tick_params(axis='y',
                direction='out', 
                length=10,    
                width=3,     
                colors='black')  

plt.grid(False, which='both') 

font_properties = fm.FontProperties(weight='bold', size=28)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=False, prop=font_properties)

plt.tight_layout()

violin_plot_path = f'{output_directory}\\Flowering duration_by_year_violin_plot.png'
plt.savefig(violin_plot_path)

plt.close()

