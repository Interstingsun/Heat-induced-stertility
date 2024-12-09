import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.font_manager as fm 
import os

mpl.rcParams['font.family'] = 'Times New Roman'
current_dir = os.path.dirname(os.path.realpath(__file__))

file_path = os.path.join(current_dir, '2022_2023_Flowering_distribution_Sterility.xlsx')
df = pd.read_excel(file_path)

df_clean = df.dropna(subset=['Simulated cumulative sterility (optimum)', 'Observed sterility', 'Temperature threshold'])

max_simulated_sterility = df_clean.groupby(['Variety', 'Year'], as_index=False)['Simulated cumulative sterility (optimum)'].max()

result = pd.merge(max_simulated_sterility, df_clean[['Variety', 'Year', 'Simulated cumulative sterility (optimum)', 'Observed sterility', 'Temperature threshold']], 
                  on=['Variety', 'Year', 'Simulated cumulative sterility (optimum)'])

result = result[['Variety', 'Year', 'Simulated cumulative sterility (optimum)', 'Observed sterility', 'Temperature threshold']]

plt.figure(figsize=(10, 10))

plt.xlim(0, 0.6)
plt.ylim(0, 0.6)
plt.xticks(np.arange(0, 0.7, 0.1))
plt.yticks(np.arange(0, 0.7, 0.1))

within_range = (result['Simulated cumulative sterility (optimum)'] * 0.8 <= result['Observed sterility']) & (result['Observed sterility'] <= result['Simulated cumulative sterility (optimum)'] * 1.2)
outside_range = ~within_range

plt.scatter(result[within_range]['Observed sterility'], result[within_range]['Simulated cumulative sterility (optimum)'], c=(0/255, 128/255, 128/255), edgecolor='none', label='Within ±20% error', s=250, alpha=0.6)
plt.scatter(result[outside_range]['Observed sterility'], result[outside_range]['Simulated cumulative sterility (optimum)'], c='red', edgecolor='none', label='Outside ±20% error', s=200, alpha=0.3)

max_value = max(result['Observed sterility'].max(), result['Simulated cumulative sterility (optimum)'].max())
plt.plot([0, 0.6], [0, 0.6], color='black', linestyle='--', label='1:1 line', linewidth=3)

error_x1 = np.linspace(0.5, 0.5, 2)
error_x2 = np.linspace(0.6, 0.6, 2)

plt.plot([0, error_x1[0]], [0, error_x1[0] * 1.2], color='red', alpha=0.3, linestyle=':', label='±20% error line', linewidth=3)
plt.plot([0, error_x2[0]], [0, error_x2[0] * 0.8], color='red', alpha=0.3, linestyle=':', linewidth=3)

plt.xlabel('Observed sterility', fontsize=32, color='black', weight='bold', labelpad=25)
plt.ylabel('Simulated sterility', fontsize=32, color='black', weight='bold', labelpad=25)

plt.xticks(fontsize=24, color='black', weight='bold')
plt.yticks(fontsize=24, color='black', weight='bold')

ax = plt.gca()
ax.set_facecolor('none')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)
    spine.set_linestyle('-') 
plt.tick_params(axis='x', direction='out', length=10, width=3, colors='black')
plt.tick_params(axis='y', direction='out', length=10, width=3, colors='black')

plt.grid(False, which='both')

font_properties = fm.FontProperties(weight='bold', size=28)
plt.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95), frameon=False, prop=font_properties)

plt.tight_layout()

save_path = os.path.join(current_dir, 'Observed_vs_Simulated_Sterility.png')
plt.savefig(save_path, bbox_inches='tight')

plt.show()

result['In ±20% Error Range'] = result.apply(lambda row: 'Yes' if row['Simulated cumulative sterility (optimum)'] * 0.8 <= row['Observed sterility'] <= row['Simulated cumulative sterility (optimum)'] * 1.2 else 'No', axis=1)

marked_output_path =  os.path.join(current_dir, 'Observed_vs_Simulated_Sterility.xlsx')
result.to_excel(marked_output_path, index=False)
