import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
import matplotlib.font_manager as fm 
from matplotlib.dates import DateFormatter, MonthLocator
import os

mpl.rcParams['font.family'] = 'Times New Roman'

current_dir = os.path.dirname(os.path.realpath(__file__))

file_path = os.path.join(current_dir, '2022_2023_meteorology.xlsx')
output_image_path = os.path.join(current_dir, 'Meteorology.png')

df = pd.read_excel(file_path)

df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

df['MAX'] = (df['MAX'] - 32) / 1.8
df['MIN'] = (df['MIN'] - 32) / 1.8

df['Month'] = df['DATE'].dt.month
df['Day'] = df['DATE'].dt.day

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

font_properties = fm.FontProperties(weight='bold', size=28)

year_2022_data = df[df['DATE'].dt.year == 2022]
axes[0].plot(year_2022_data['Month'] + (year_2022_data['Day'] / 31), year_2022_data['MAX'], label='Daily maximum temperature', color=(250/255, 60/255, 60/255), linewidth=2)
axes[0].plot(year_2022_data['Month'] + (year_2022_data['Day'] / 31), year_2022_data['MIN'], label='Daily minimum temperature', linestyle='--', color=(0/255, 128/255, 128/255), linewidth=2)
axes[0].text(0.5, 0.94, '2022', fontsize=24, ha='center', fontweight='bold',  va='center', transform=axes[0].transAxes)
for label in axes[0].get_yticklabels():
    label.set_fontweight('bold')
axes[0].set_ylabel('Temperature (°C)', fontproperties=font_properties,  labelpad=20)
axes[0].set_ylim(5, 45)
axes[0].set_yticks(range(5, 46, 10))
axes[0].set_xlim(5, 11)
axes[0].set_xticks(range(5, 11, 1))
axes[0].tick_params(axis='x', labelsize=24, labelcolor='black', length=10, width=3)  
axes[0].tick_params(axis='y', labelsize=24, labelcolor='black', length=10, width=3) 
axes[0].xaxis.label.set_fontweight('bold') 
axes[0].yaxis.label.set_fontweight('bold') 
axes[0].grid(False)
font_properties = fm.FontProperties(weight='bold', size=20)
axes[0].legend(frameon=False, loc='lower center', labelspacing=0.5, prop=font_properties)

ax = axes[0]
ax.set_facecolor('none')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)
    spine.set_linestyle('-')

plt.tick_params(axis='x', direction='out', length=10, width=3, colors='black')
plt.tick_params(axis='y', direction='out', length=10, width=3, colors='black')
plt.grid(False, which='both')

font_properties = fm.FontProperties(weight='bold', size=28)
year_2023_data = df[df['DATE'].dt.year == 2023]
axes[1].plot(year_2023_data['Month'] + (year_2023_data['Day'] / 31), year_2023_data['MAX'], label='MAX 2023', color=(250/255, 60/255, 60/255),  linewidth=2)
axes[1].plot(year_2023_data['Month'] + (year_2023_data['Day'] / 31), year_2023_data['MIN'], label='MIN 2023', linestyle='--', color=(0/255, 128/255, 128/255),  linewidth=2)
axes[1].text(0.5, 0.90, '2023', fontsize=24, ha='center', fontweight='bold', va='center', transform=axes[1].transAxes)
for label in axes[1].get_yticklabels():
    label.set_fontweight('bold')
axes[1].set_xlabel('Month', fontproperties=font_properties,  labelpad=20)
axes[1].set_ylabel('Temperature (°C)', fontproperties=font_properties,  labelpad=20)
axes[1].set_ylim(5, 45)
axes[1].set_yticks(range(5, 46, 10))
axes[1].set_xlim(5, 11)
axes[1].set_xticks(range(5, 11, 1))
axes[1].tick_params(axis='x', labelsize=24, labelcolor='black',  length=10, width=3)  
axes[1].tick_params(axis='y', labelsize=24, labelcolor='black',  length=10, width=3) 
axes[1].xaxis.label.set_fontweight('bold') 
axes[1].yaxis.label.set_fontweight('bold')
axes[1].grid(False)

ax = axes[1]
ax.set_facecolor('none')
for spine in ax.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3)
    spine.set_linestyle('-')

plt.tick_params(axis='x', direction='out', length=10, width=3, colors='black')
plt.tick_params(axis='y', direction='out', length=10, width=3, colors='black')

font_props = font_manager.FontProperties(weight='bold', size=24)
axes[0].set_xticks(np.arange(5, 12))
axes[0].set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'], fontproperties=font_props)
axes[1].set_xticks(np.arange(5, 12))
axes[1].set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov'], fontproperties=font_props)

plt.subplots_adjust(hspace=0)

plt.savefig(output_image_path, dpi=600, bbox_inches='tight')

plt.show()
