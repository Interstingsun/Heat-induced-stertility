import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, 'Observed_vs_Simulated_Sterility.xlsx')
df = pd.read_excel(file_path)

valid_data = df[df['In ±20% Error Range'] == 'Yes']
valid_data = valid_data.drop_duplicates(subset=['Variety'])
valid_data['Heat_resistance_level'] = pd.qcut(valid_data['Temperature threshold'], q=5, labels=[1, 2, 3, 4, 5])

final_df = valid_data[['Variety', 'Temperature threshold', 'Heat_resistance_level']].sort_values(by='Heat_resistance_level', ascending=False)
output_file = os.path.join(current_dir, 'Heat_resistance_level_multi_year_data.xlsx')
final_df.to_excel(output_file, index=False)

final_df = pd.read_excel(output_file)
final_df = final_df.sort_values(by=['Heat_resistance_level', 'Temperature threshold'], ascending=[False, False])

plt.style.use('seaborn-white')

cmap = plt.cm.Spectral
norm = mpl.colors.Normalize(vmin=final_df['Heat_resistance_level'].min(), vmax=final_df['Heat_resistance_level'].max())

plt.figure(figsize=(20, 10))
bars = plt.bar(final_df['Variety'], final_df['Temperature threshold'], color=cmap(norm(final_df['Heat_resistance_level'])))

plt.xlabel('Variety', fontsize=28, color='black', weight='bold', labelpad=15, family='Times New Roman')
plt.ylabel(r'Temperature threshold (°C)', fontsize=28, color='black', weight='bold', labelpad=15, family='Times New Roman')

plt.xticks(fontsize=24, color='black', rotation=90, ha='center', family='Times New Roman')
plt.yticks(fontsize=24, color='black', family='Times New Roman')

plt.xlim(-1, len(final_df['Variety']))
plt.ylim(33, 45)

ax = plt.gca()
ax.set_facecolor('white')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)
    spine.set_linestyle('-')

plt.grid(False)

plt.tick_params(axis='x', direction='out', length=10, width=3, colors='black', grid_color='black', labelsize=24, pad=15)
plt.tick_params(axis='y', direction='out', length=10, width=3, colors='black', grid_color='black', labelsize=24, pad=15)

handles, labels = [], []
for level in range(1, 6):
    color = cmap(norm(level))
    handles.append(plt.Line2D([0], [0], marker='s', color='w', label=f'Level {level}', markerfacecolor=color, markersize=20))
    labels.append(f'Level {level}')
plt.legend(handles=handles, title='Heat resistance level', loc='upper right', frameon=False,  
           bbox_to_anchor=(0.9, 0.9), prop=fm.FontProperties(family='Times New Roman', size=28, weight='bold'),
           title_fontproperties=fm.FontProperties(family='Times New Roman', size=28, weight='bold'))

output_image_path =os.path.join(current_dir, 'Heat_resistance_level_multi_year_data.png')
plt.savefig(output_image_path, dpi=600, bbox_inches='tight')

plt.tight_layout()
plt.show()
