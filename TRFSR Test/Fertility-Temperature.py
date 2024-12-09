import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import font_manager
import matplotlib.font_manager as fm 
import os

mpl.rcParams['font.family'] = 'Times New Roman'
current_dir = os.path.dirname(os.path.realpath(__file__))

phenology_df = pd.read_excel(os.path.join(current_dir, '2022_2023_phenology_fertility.xlsx'))
meteorology_df = pd.read_excel(os.path.join(current_dir, '2022_2023_meteorology.xlsx'))

phenology_df.columns = phenology_df.columns.str.strip()

heading_dates = phenology_df[['Variety', '10%heading', '80%heading', 'Fertility']]

heading_dates['10%heading'] = pd.to_datetime(heading_dates['10%heading'])
heading_dates['80%heading'] = pd.to_datetime(heading_dates['80%heading'])

average_temperatures = []
average_daytime_temperatures = []
average_mintemperatures = []

for _, row in heading_dates.iterrows():
    fertility_value = row.get('Fertility', None)
    if pd.isna(fertility_value):
        average_temperatures.append(None)
        average_daytime_temperatures.append(None)
        average_mintemperatures.append(None)
    else:
        start_date = row['10%heading']
        end_date = row['80%heading']
        
        meteorology_df['DATE'] = pd.to_datetime(meteorology_df['DATE'], errors='coerce')
        
        relevant_data = meteorology_df[(meteorology_df['DATE'] >= start_date) & (meteorology_df['DATE'] <= end_date)]
        
        if 'MAX' in relevant_data.columns and 'MIN' in relevant_data.columns:
            relevant_data.loc[:, 'High_Temperature_C'] = (relevant_data['MAX'] - 32) * 5 / 9
            relevant_data.loc[:, 'MAX_C'] = (relevant_data['MAX'] - 32) * 5 / 9
            relevant_data.loc[:, 'MIN_C'] = (relevant_data['MIN'] - 32) * 5 / 9
            
            relevant_data.loc[:, 'Daytime_Temperature_C'] = 0.75 * relevant_data['MAX_C'] + 0.25 * relevant_data['MIN_C']
            
            avg_temp = relevant_data['High_Temperature_C'].mean()
            average_temperatures.append(avg_temp)
            
            avg_daytime_temp = relevant_data['Daytime_Temperature_C'].mean()
            average_daytime_temperatures.append(avg_daytime_temp)

            avg_mintemp = relevant_data['MIN_C'].mean()
            average_mintemperatures.append(avg_mintemp)
        else:
            average_temperatures.append(None)
            average_daytime_temperatures.append(None)
            average_mintemperatures.append(None)

phenology_df['Average_Temperature_C'] = average_temperatures
phenology_df['Average_Daytime_Temperature_C'] = average_daytime_temperatures
phenology_df['Average_min_Temperature_C'] = average_mintemperatures

phenology_df['Fertility'] = phenology_df['Fertility'] * 100
phenology_df = phenology_df.dropna(subset=['Fertility'])

def fertility_curve(temp):
    return np.minimum(100, np.maximum(0,  100 / (1 + np.exp(0.853 * (temp - 36.6)))))

def fertility_curve2(temp):
    return np.where(
        temp > 33, 
        100 * (( (temp - 10) / (33 - 10) ) * ((43 - temp) / (43 - 33)) ** ((43 - 33) / (33 - 10))) ** 12.5, 
        100
    )

def fertility_curve3(temp):
    return np.minimum(100, np.maximum(0,100* (42 - temp) / (42 - 35)))

def fertility_curve4(temp):
    return np.minimum(100, np.maximum(0,  100 / (1 + np.exp(0.853 * (temp - 38.8)))))

def fertility_curve_daytime(temp):
    return np.where(temp >= 28, (1 - 0.1 * (temp - 28)) * 100, 100)

temperature_range = np.linspace(28, 42, 500)
fertility_values = fertility_curve(temperature_range)
fertility_values2 = fertility_curve2(temperature_range)
fertility_values3 = fertility_curve3(temperature_range)
fertility_values4 = fertility_curve4(temperature_range)

plt.figure(figsize=(10, 10))
plt.scatter(
    phenology_df['Average_Temperature_C'], 
    phenology_df['Fertility'], 
    s=250, 
    c=(0/255, 128/255, 128/255), 
    edgecolors='None',
    alpha=0.6,
    label='Observation'
)

plt.plot(temperature_range, fertility_values, color=(250/255, 60/255, 60/255), label='Horie-type (36.6)', linewidth=5, alpha=0.8)
plt.plot(temperature_range, fertility_values2, color=(250/255, 100/255, 100/255), label='MCWLA-Rice', linewidth=5, linestyle=':', alpha=0.8)
plt.plot(temperature_range, fertility_values3, color=(250/255, 100/255, 100/255), label='SAMARA', linewidth=5, linestyle='--', alpha=0.8)
plt.plot(temperature_range, fertility_values4, color=(250/255, 100/255, 100/255), label='GEMRICE (38.8)', linewidth=5, linestyle='-.', alpha=0.8)

plt.xlabel(r'Average daily maximum temperature (°C)', fontsize=28, color='black', weight='bold', labelpad=15)
plt.ylabel(r'Seed-setting rate (%)', fontsize=28, color='black', weight='bold', labelpad=15)

plt.xticks(fontsize=24, color='black', weight='bold')
plt.yticks(fontsize=24, color='black', weight='bold')

plt.xlim(28, 42)
plt.ylim(0, 110)

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
plt.legend(loc='lower left', bbox_to_anchor=(0.05, 0.05), frameon=False, prop=font_properties)

output_image_path = os.path.join(current_dir, 'Fertility_Maxmium_Temperature.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

temperature_range_daytime = np.linspace(26, 38, 500)
fertility_values_daytime = fertility_curve_daytime(temperature_range_daytime)

plt.figure(figsize=(10, 10))

plt.scatter(
    phenology_df['Average_Daytime_Temperature_C'], 
    phenology_df['Fertility'], 
    s=250, 
    c=(0/255, 128/255, 128/255), 
    edgecolors='none',
    alpha=0.6,
    label='Observation'
)

plt.plot(
    temperature_range_daytime,
    fertility_values_daytime,
    color=(250/255, 60/255, 60/255), 
    label='CERES-Rice',
    linewidth=5, 
    alpha=0.8
)

plt.xlabel(r'Average daytime temperature (°C)', fontsize=28, color='black', weight='bold', labelpad=15)
plt.ylabel(r'Seed-setting rate (%)', fontsize=28, color='black', weight='bold', labelpad=15)

plt.xticks(fontsize=24, color='black', weight='bold')
plt.yticks(fontsize=24, color='black', weight='bold')

plt.xlim(26, 38)
plt.ylim(0, 110)

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
plt.legend(loc='lower left', bbox_to_anchor=(0.1, 0.2), frameon=False, prop=font_properties)

output_image_path = os.path.join(current_dir, 'Fertility_Daytime_Temperature.png')
plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

plt.show()

