import os 
import numpy as np 
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib import font_manager
import matplotlib.ticker as ticker


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, '2022_2023_phenology_fertility.xlsx')
df = pd.read_excel(file_path)

weather_file_path = os.path.join(current_dir, '2022_2023_meteorology.xlsx')
weather_df = pd.read_excel(weather_file_path)

weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

weather_df['Year'] = weather_df['DATE'].dt.year
df['Year'] = pd.to_datetime(df['10%heading']).dt.year

result = []

def calculate_optimal_m(start_prob, end_prob, start_day, end_day):
    def error(m):
        cdf_start = poisson.cdf(start_day, m)  
        cdf_end = poisson.cdf(end_day, m) 
        return abs(cdf_start - start_prob) + abs(cdf_end - end_prob)
    
    result = minimize_scalar(error, bounds=(0, 100), method='bounded')
    return result.x

def calculate_sterility(temp, threshold=36.6):
    return 1 - max(0, min(1, 1 / (1 + np.exp(0.853 * (temp - threshold)))))


def calculate_cumulative_sterility(result_subset):
    result_subset['Weighted_sterility'] = result_subset['Simulated daily sterility'] * result_subset['Poisson probability']
    cumulative_sterility = result_subset['Weighted_sterility'].sum()
    return cumulative_sterility


def calculate_error(threshold, variety, result_df, Ob_Sterility_df):
    result_subset = result_df[result_df['Variety'] == variety]
    
    result_subset['Simulated daily sterility'] = result_subset['MAX (°C)'].apply(calculate_sterility, threshold=threshold)
    
    cumulative_sterility = calculate_cumulative_sterility(result_subset)
    
    observed_sterility = Ob_Sterility_df[Ob_Sterility_df['Variety'] == variety]['Observed sterility'].values
    
    if len(observed_sterility) == 0:
        return np.inf 
    observed_at_max_prob = observed_sterility[0] 
    
    error = abs(cumulative_sterility - observed_at_max_prob)
    return error

def find_optimal_threshold(variety, result_df, Ob_Sterility_df):
    if variety not in Ob_Sterility_df['Variety'].values:
        return np.nan
    
    optimal_threshold = minimize_scalar(calculate_error, bounds=(15, 50), args=(variety, result_df, Ob_Sterility_df), method='bounded').x
    return optimal_threshold


for index, row in df.iterrows():
    variety = row['Variety']
    heading_start = pd.to_datetime(row['10%heading'])
    heading_end = pd.to_datetime(row['80%heading'])
    year = heading_start.year
    
    days_in_range = (heading_end - heading_start).days
    
    target_start_prob = 0.10
    target_end_prob = 0.80

    m_optimal = calculate_optimal_m(target_start_prob, target_end_prob, 0, days_in_range)

    cumulative_prob = 0
    daily_probabilities = []
    j = 0
    while poisson.cdf(j, m_optimal) < 0.001:
        j += 1
    
    cumulative_prob = poisson.cdf(j, m_optimal)
    date = heading_start - timedelta(days=j)  
    while cumulative_prob < 0.999:  
        prob = poisson.pmf(j, m_optimal) 
        cumulative_prob += prob
        daily_probabilities.append({
            'Variety': variety,
            'Year': year,
            'DATE': date,
            'Poisson probability': prob,
            'Cumulative probability': cumulative_prob
        })
        j += 1
        date += timedelta(days=1)
    
    result.extend(daily_probabilities)

result_df = pd.DataFrame(result)

result_df['DATE'] = pd.to_datetime(result_df['DATE'])

weather_df['MAX'] = (weather_df['MAX'] - 32) * 5 / 9 
weather_df.rename(columns={'MAX':'MAX (°C)'}, inplace=True)
result_df = pd.merge(result_df, weather_df[['DATE','MAX (°C)', 'Year']], on=['DATE', 'Year'], how='left')

Ob_Fertility_file_path = os.path.join(current_dir, '2022_2023_phenology_fertility.xlsx')
Ob_Fertility_df = pd.read_excel(Ob_Fertility_file_path)

Ob_Fertility_df = Ob_Fertility_df[['Variety', 'Fertility', '80%heading']]

Ob_Fertility_df['Observed sterility'] = 1 - Ob_Fertility_df['Fertility']

Ob_Fertility_df['Year'] = pd.to_datetime(Ob_Fertility_df['80%heading']).dt.year

Ob_Sterility_df = Ob_Fertility_df[['Variety', 'Year', 'Observed sterility']]

result_df = pd.merge(result_df, Ob_Sterility_df, on=['Variety', 'Year'], how='left')

optimal_thresholds = {}
for variety in df['Variety'].unique():
    if Ob_Sterility_df[Ob_Sterility_df['Variety'] == variety]['Observed sterility'].isna().any():
        optimal_thresholds[variety] = np.nan
    else:
        optimal_thresholds[variety] = find_optimal_threshold(variety, result_df, Ob_Sterility_df)

result_df['Temperature threshold'] = result_df['Variety'].apply(lambda variety: optimal_thresholds.get(variety, np.nan))


def calculate_sterility_with_optimal_threshold(temp, optimal_threshold):
    if np.isnan(optimal_threshold):
        return np.nan
    return 1 - max(0, min(1, 1 / (1 + np.exp(0.853 * (temp - optimal_threshold)))))

def calculate_daily_cumulative_sterility_with_optimal_threshold(result_subset, optimal_threshold):
    if np.isnan(optimal_threshold):
        return np.nan
    
    result_subset['Simulated daily sterility (optimum)'] = result_subset['MAX (°C)'].apply(
        calculate_sterility_with_optimal_threshold, optimal_threshold=optimal_threshold)
    result_subset['Weighted sterility (optimum)'] = result_subset['Simulated daily sterility (optimum)'] * result_subset['Poisson probability']

    result_subset['Simulated cumulative sterility (optimum)'] = result_subset['Weighted sterility (optimum)'].cumsum()
    return result_subset

for variety in df['Variety'].unique():
    for year in df['Year'].unique():
        optimal_threshold = optimal_thresholds.get(variety, np.nan)
        
        result_subset = result_df[(result_df['Variety'] == variety) & (result_df['Year'] == year)]

        if not np.isnan(optimal_threshold):
            result_subset = calculate_daily_cumulative_sterility_with_optimal_threshold(result_subset, optimal_threshold)
            
            result_df.loc[(result_df['Variety'] == variety) & (result_df['Year'] == year), 'Simulated daily sterility (optimum)'] = result_subset['Simulated daily sterility (optimum)']
            result_df.loc[(result_df['Variety'] == variety) & (result_df['Year'] == year), 'Simulated cumulative sterility (optimum)'] = result_subset['Simulated cumulative sterility (optimum)']

for variety in df['Variety'].unique():
    for year in df['Year'].unique():
        optimal_threshold = optimal_thresholds.get(variety, np.nan)
        result_subset = result_df[(result_df['Variety'] == variety) & (result_df['Year'] == year)]
        
        if not np.isnan(optimal_threshold):
            result_subset = calculate_daily_cumulative_sterility_with_optimal_threshold(result_subset, optimal_threshold)

            result_df.loc[(result_df['Variety'] == variety) & (result_df['Year'] == year), 'Simulated daily sterility (optimum)'] = result_subset['Simulated daily sterility (optimum)']
            result_df.loc[(result_df['Variety'] == variety) & (result_df['Year'] == year), 'Simulated cumulative sterility (optimum)'] = result_subset['Simulated cumulative sterility (optimum)']
            
            result_df.loc[(result_df['Variety'] == variety) & (result_df['Year'] == year), 'Temperature threshold'] = optimal_threshold


output_file_path = os.path.join(current_dir, '2022_2023_Flowering_distribution_Sterility.xlsx')

result_df.to_excel(output_file_path, index=False)


unique_years = result_df['Year'].unique()
num_years = len(unique_years)

fig, axes = plt.subplots(num_years, 4, figsize=(16, 4 * num_years))

colors = plt.cm.tab10(np.linspace(0, 1, result_df['Variety'].nunique()))

for i, year in enumerate(unique_years):
    year_df = result_df[result_df['Year'] == year]
    
    date_locator = mdates.AutoDateLocator()
    date_formatter = mdates.DateFormatter('%m-%d')
    
    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 0].plot(subset['DATE'], subset['Poisson probability'], label=variety, color=colors[j])
    axes[i, 0].set_title(f'{year} - Poisson probability')
    axes[i, 0].set_xlabel('DATE')
    axes[i, 0].set_ylabel('Poisson probability')
    axes[i, 0].xaxis.set_major_locator(date_locator)
    axes[i, 0].xaxis.set_major_formatter(date_formatter)
    axes[i, 0].tick_params(axis='x', rotation=45)  
    

    axes[i, 0].autoscale(enable=True, axis='y')

    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 1].plot(subset['DATE'], subset['Cumulative probability'], label=variety, color=colors[j])
    axes[i, 1].set_title(f'{year} - Cumulative probability')
    axes[i, 1].set_xlabel('DATE')
    axes[i, 1].set_ylabel('Cumulative probability')
    axes[i, 1].xaxis.set_major_locator(date_locator)
    axes[i, 1].xaxis.set_major_formatter(date_formatter)
    axes[i, 1].tick_params(axis='x', rotation=45)  
    axes[i, 1].autoscale(enable=True, axis='y')

    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 2].plot(subset['DATE'], subset['Simulated daily sterility (optimum)'], label=variety, color=colors[j])
    axes[i, 2].set_title(f'{year} - Simulated daily sterility')
    axes[i, 2].set_xlabel('Date')
    axes[i, 2].set_ylabel('Simulated daily sterility')
    axes[i, 2].xaxis.set_major_locator(date_locator)
    axes[i, 2].xaxis.set_major_formatter(date_formatter)
    axes[i, 2].tick_params(axis='x', rotation=45)  

    axes[i, 2].autoscale(enable=True, axis='y')

    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 3].plot(subset['DATE'], subset['Simulated cumulative sterility (optimum)'], label=variety, color=colors[j])
    axes[i, 3].set_title(f'{year} - Simulated cumulative sterility')
    axes[i, 3].set_xlabel('Date')
    axes[i, 3].set_ylabel('Simulated cumulative sterility')
    axes[i, 3].xaxis.set_major_locator(date_locator)
    axes[i, 3].xaxis.set_major_formatter(date_formatter)
    axes[i, 3].tick_params(axis='x', rotation=45) 

    axes[i, 3].autoscale(enable=True, axis='y')


plt.tight_layout(h_pad=2.0, w_pad=1.0)

output_path = os.path.join(current_dir)
os.makedirs(output_path, exist_ok=True)

date_locator = mdates.DayLocator(interval=5) 
date_formatter = mdates.DateFormatter('%m-%d') 

english_font = {'family': 'Times New Roman'} 

for i, year in enumerate(unique_years):
    for j, title in enumerate(['Poisson probability', 'Cumulative probability', 'Simulated daily sterility (optimum)', 'Simulated cumulative sterility (optimum)']):
        fig_single, ax_single = plt.subplots(figsize=(10, 10))
        
        year_df = result_df[result_df['Year'] == year]
        for k, variety in enumerate(year_df['Variety'].unique()):
            subset = year_df[year_df['Variety'] == variety]
            ax_single.plot(subset['DATE'], subset[title], label=variety, color=colors[k])
        
        ax_single.text(0.5, 0.95, f'{year}', transform=ax_single.transAxes, fontsize=28,
                       fontweight='bold', fontproperties=english_font, ha='center')
        ax_single.set_xlabel('Date', fontsize=32, fontweight='bold', fontproperties=english_font, labelpad=25)
        ax_single.set_ylabel(title, fontsize=32, fontweight='bold', fontproperties=english_font, labelpad=25)
        ax_single.xaxis.set_major_locator(date_locator)
        ax_single.xaxis.set_major_formatter(date_formatter)
        ax_single.tick_params(axis='x', rotation=45)
        ax_single.tick_params(axis='both', labelcolor='black', width=2, length=12, direction='out')
        for label in ax_single.get_xticklabels() + ax_single.get_yticklabels():
            label.set_fontsize(24)
            label.set_fontweight('bold')
            label.set_family('Times New Roman')
        for spine in ax_single.spines.values():
            spine.set_linewidth(2)  


        if title == 'Poisson probability':
            ax_single.set_ylim(0, 0.25)
            ax_single.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        elif title == 'Cumulative probability':
            ax_single.set_ylim(0, 1.1)
            ax_single.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        elif title == 'Simulated daily sterility (optimum)':
            ax_single.set_ylim(0, 1.1)
            ax_single.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        elif title == 'Simulated cumulative sterility (optimum)':
            ax_single.set_ylim(0, 0.6)
            ax_single.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

        output_file = os.path.join(output_path, f'{year}_{title}.png')
        fig_single.savefig(output_file, dpi=600, bbox_inches='tight') 
        plt.close(fig_single) 
        print(f"Figures have saved to: {output_file}")

plt.show()