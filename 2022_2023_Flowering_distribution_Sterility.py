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


# 设置英文字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# 读取结实率数据
file_path = r'C:\Users\sunti\Desktop\Fertility\Multi_year\En\2022_2023_phenology_fertility.xlsx'
df = pd.read_excel(file_path)

# 读取气象数据
weather_file_path = r'C:\Users\sunti\Desktop\Fertility\Multi_year\En\2022_2023_meteorology.xlsx'
weather_df = pd.read_excel(weather_file_path)

# 确保日期列是 datetime 类型
weather_df['DATE'] = pd.to_datetime(weather_df['DATE'])

# 添加年份列
weather_df['Year'] = weather_df['DATE'].dt.year
df['Year'] = pd.to_datetime(df['10%heading']).dt.year  # 假设始穗期列的年份可以代表品种年份

# 初始化结果列表
result = []

# 计算泊松分布的参数m，使得Cumulative probability尽可能接近目标
def calculate_optimal_m(start_prob, end_prob, start_day, end_day):
    """计算最优m使得泊松分布Cumulative probability分别接近start_prob和end_prob"""
    def error(m):
        """计算误差：始穗期Cumulative probability与目标start_prob的差值 + 齐穗期Cumulative probability与目标end_prob的差值"""
        cdf_start = poisson.cdf(start_day, m)  # 始穗期的Cumulative probability
        cdf_end = poisson.cdf(end_day, m)  # 齐穗期的Cumulative probability
        return abs(cdf_start - start_prob) + abs(cdf_end - end_prob)
    
    # 使用最小化误差的方式来寻找最优的m值
    result = minimize_scalar(error, bounds=(0, 100), method='bounded')
    return result.x

# 计算Simulated daily sterility
def calculate_sterility(temp, threshold=36.6):
    """根据气温计算每日不育率"""
    return 1 - max(0, min(1, 1 / (1 + np.exp(0.853 * (temp - threshold)))))


# 计算Simulated cumulative sterility
def calculate_cumulative_sterility(result_subset):
    """计算Simulated cumulative sterility（加权和）"""
    result_subset['Weighted_sterility'] = result_subset['Simulated daily sterility'] * result_subset['Poisson probability']
    cumulative_sterility = result_subset['Weighted_sterility'].sum()  # Simulated cumulative sterility是加权和
    return cumulative_sterility


# 计算误差：不育率模拟值与观测值的误差
def calculate_error(threshold, variety, result_df, Ob_Sterility_df):
    """计算不育率模拟值与不育率观测值的误差"""
    # 获取该品种的结果子集
    result_subset = result_df[result_df['Variety'] == variety]
    
    # 计算Simulated daily sterility
    result_subset['Simulated daily sterility'] = result_subset['MAX (°C)'].apply(calculate_sterility, threshold=threshold)
    
    # 计算Simulated cumulative sterility
    cumulative_sterility = calculate_cumulative_sterility(result_subset)
    
    # 获取不育率观测值
    observed_sterility = Ob_Sterility_df[Ob_Sterility_df['Variety'] == variety]['Observed sterility'].values
    
    # 如果该品种没有不育率观测值，则跳过
    if len(observed_sterility) == 0:
        return np.inf  # 返回一个大的误差，表示跳过
    
    # 提取最大Cumulative probability时对应的不育率观测值
    observed_at_max_prob = observed_sterility[0]  # 直接提取该品种的不育率观测值
    
    # 计算误差：模拟值和观测值的差异
    error = abs(cumulative_sterility - observed_at_max_prob)
    return error


def find_optimal_threshold(variety, result_df, Ob_Sterility_df):
    """寻找最优温度阈值使得误差最小"""
    # 检查该品种是否有不育率观测值
    if variety not in Ob_Sterility_df['Variety'].values:
        print(f"跳过品种 {variety}：缺少不育率观测值")
        return np.nan  # 返回 NaN，表示该品种没有不育率观测值
    
    # 如果有不育率观测值，进行最优温度阈值的计算
    optimal_threshold = minimize_scalar(calculate_error, bounds=(15, 50), args=(variety, result_df, Ob_Sterility_df), method='bounded').x
    return optimal_threshold


# 遍历每个品种
for index, row in df.iterrows():
    # 获取品种名，始穗期，齐穗期，年份
    variety = row['Variety']
    heading_start = pd.to_datetime(row['10%heading'])
    heading_end = pd.to_datetime(row['80%heading'])
    year = heading_start.year
    
    # 计算始穗期到齐穗期之间的天数差
    days_in_range = (heading_end - heading_start).days
    
    # 目标：Cumulative probability为10%时为始穗期，Cumulative probability为80%时为齐穗期
    target_start_prob = 0.10  # 始穗期累积10%
    target_end_prob = 0.80  # 齐穗期累积80%

    # 计算最优的泊松分布参数m
    m_optimal = calculate_optimal_m(target_start_prob, target_end_prob, 0, days_in_range)

    # 从Cumulative probability达到0.1%处开始输出每日概率，直到Cumulative probability达到99.9%
    cumulative_prob = 0
    daily_probabilities = []
    j = 0
    # 找到Cumulative probability为0.1%的起点
    while poisson.cdf(j, m_optimal) < 0.001:
        j += 1
    
    # 从找到的累积0.1%处开始计算
    cumulative_prob = poisson.cdf(j, m_optimal)
    date = heading_start - timedelta(days=j)  # 设定初始日期向前拓展至0.1%对应的日期
    while cumulative_prob < 0.999:  # Cumulative probability达到99.9%时停止
        prob = poisson.pmf(j, m_optimal)  # 第j天的Poisson概率
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
    
    # 将每日的结果加入到最终结果中
    result.extend(daily_probabilities)

# 将结果转换为DataFrame
result_df = pd.DataFrame(result)

# 确保日期列是 datetime 类型
result_df['DATE'] = pd.to_datetime(result_df['DATE'])

# 合并气象数据：获取日期和最高气温（转换为℃）
weather_df['MAX'] = (weather_df['MAX'] - 32) * 5 / 9  # 转换为摄氏度
weather_df.rename(columns={'MAX':'MAX (°C)'}, inplace=True)  # 修改列名为“最高气温(°C)”
result_df = pd.merge(result_df, weather_df[['DATE','MAX (°C)', 'Year']], on=['DATE', 'Year'], how='left')

# 读取结实率数据
Ob_Fertility_file_path = r'C:\Users\sunti\Desktop\Fertility\Multi_year\En\2022_2023_phenology_fertility.xlsx'
Ob_Fertility_df = pd.read_excel(Ob_Fertility_file_path)

# 确保"结实率"列与品种列对应
Ob_Fertility_df = Ob_Fertility_df[['Variety', 'Fertility', '80%heading']]

# 计算不育率观测值：1 - 结实率
Ob_Fertility_df['Observed sterility'] = 1 - Ob_Fertility_df['Fertility']

# 从"齐穗期"提取年份
Ob_Fertility_df['Year'] = pd.to_datetime(Ob_Fertility_df['80%heading']).dt.year

# 获取不育率观测值数据
Ob_Sterility_df = Ob_Fertility_df[['Variety', 'Year', 'Observed sterility']]  # 获取包含品种、年份、不育率观测值的数据

# 合并数据：将观测到的结实率加入到结果
result_df = pd.merge(result_df, Ob_Sterility_df, on=['Variety', 'Year'], how='left')

# 为每个品种计算最优温度阈值
optimal_thresholds = {}
for variety in df['Variety'].unique():
    if Ob_Sterility_df[Ob_Sterility_df['Variety'] == variety]['Observed sterility'].isna().any():
        optimal_thresholds[variety] = np.nan  # 为该品种返回NaN
    else:
        optimal_thresholds[variety] = find_optimal_threshold(variety, result_df, Ob_Sterility_df)

# 将最优温度阈值添加到result_df中
result_df['Temperature threshold'] = result_df['Variety'].apply(lambda variety: optimal_thresholds.get(variety, np.nan))


# 计算Simulated daily sterility，基于最优温度阈值
def calculate_sterility_with_optimal_threshold(temp, optimal_threshold):
    """根据最优温度阈值计算Simulated daily sterility"""
    if np.isnan(optimal_threshold):
        return np.nan  # 如果最优温度阈值为空，返回NaN
    return 1 - max(0, min(1, 1 / (1 + np.exp(0.853 * (temp - optimal_threshold)))))

# 逐日累积计算Simulated cumulative sterility（最优温度阈值）
def calculate_daily_cumulative_sterility_with_optimal_threshold(result_subset, optimal_threshold):
    """计算逐日累积的不育率模拟值，使用最优温度阈值"""
    if np.isnan(optimal_threshold):
        return np.nan  # 如果最优温度阈值为空，返回 NaN
    
    # 计算Simulated daily sterility（最优阈值）
    result_subset['Simulated daily sterility (optimum)'] = result_subset['MAX (°C)'].apply(
        calculate_sterility_with_optimal_threshold, optimal_threshold=optimal_threshold)
    result_subset['Weighted sterility (optimum)'] = result_subset['Simulated daily sterility (optimum)'] * result_subset['Poisson probability']

    # 逐日累积不育率
    result_subset['Simulated cumulative sterility (optimum)'] = result_subset['Weighted sterility (optimum)'].cumsum()
    return result_subset

# 对每个品种计算逐日Simulated cumulative sterility（最优阈值）
for variety in df['Variety'].unique():
    # 获取该品种的最优温度阈值
    optimal_threshold = optimal_thresholds.get(variety, np.nan)
    
    # 获取该品种的结果子集
    result_subset = result_df[result_df['Variety'] == variety]
    
    # 逐日计算Simulated cumulative sterility（最优温度阈值），并更新 result_df
    if not np.isnan(optimal_threshold):
        result_subset = calculate_daily_cumulative_sterility_with_optimal_threshold(result_subset, optimal_threshold)
        result_df.loc[result_df['Variety'] == variety, 'Simulated daily sterility (optimum)'] = result_subset['Simulated daily sterility (optimum)']
        # 将逐日累积值写回到 result_df 中
        result_df.loc[result_df['Variety'] == variety, 'Simulated cumulative sterility (optimum)'] = result_subset['Simulated cumulative sterility (optimum)']

# 逐日累积计算Simulated cumulative sterility（最优温度阈值）并绘制图
for variety in df['Variety'].unique():
    optimal_threshold = optimal_thresholds.get(variety, np.nan)
    result_subset = result_df[result_df['Variety'] == variety]
    
    if not np.isnan(optimal_threshold):
        result_subset = calculate_daily_cumulative_sterility_with_optimal_threshold(result_subset, optimal_threshold)
        result_df.loc[result_df['Variety'] == variety, 'Simulated daily sterility (optimum)'] = result_subset['Simulated daily sterility (optimum)']
        result_df.loc[result_df['Variety'] == variety, 'Simulated cumulative sterility (optimum)'] = result_subset['Simulated cumulative sterility (optimum)']
        result_df['Temperature threshold'] = result_df['Variety'].apply(lambda x: optimal_thresholds.get(x, np.nan))

# 设置输出路径
output_file_path = r'C:\Users\sunti\Desktop\Fertility\Multi_year\En\2022_2023_Flowering_distribution_Sterility.xlsx'

# 将结果写入Excel
result_df.to_excel(output_file_path, index=False)


# 获取唯一年份数目
unique_years = result_df['Year'].unique()
num_years = len(unique_years)

# 设置子图的布局，假设我们想要 2 行多列的布局
fig, axes = plt.subplots(num_years, 4, figsize=(16, 4 * num_years))

# 颜色，用于区分不同品种
colors = plt.cm.tab10(np.linspace(0, 1, result_df['Variety'].nunique()))

# 循环每个年份，创建独立的子图
for i, year in enumerate(unique_years):
    year_df = result_df[result_df['Year'] == year]
    
    # 定义日期格式
    date_locator = mdates.AutoDateLocator()
    date_formatter = mdates.DateFormatter('%m-%d')  # 只显示月-日
    
    # 绘制Poisson probability
    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 0].plot(subset['DATE'], subset['Poisson probability'], label=variety, color=colors[j])
    axes[i, 0].set_title(f'{year} - Poisson probability')
    axes[i, 0].set_xlabel('DATE')
    axes[i, 0].set_ylabel('Poisson probability')
    axes[i, 0].xaxis.set_major_locator(date_locator)
    axes[i, 0].xaxis.set_major_formatter(date_formatter)
    axes[i, 0].tick_params(axis='x', rotation=45)  # 自动旋转X轴日期
    
    # 自动调整Y轴范围
    axes[i, 0].autoscale(enable=True, axis='y')

    # 绘制Cumulative probability
    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 1].plot(subset['DATE'], subset['Cumulative probability'], label=variety, color=colors[j])
    axes[i, 1].set_title(f'{year} - Cumulative probability')
    axes[i, 1].set_xlabel('DATE')
    axes[i, 1].set_ylabel('Cumulative probability')
    axes[i, 1].xaxis.set_major_locator(date_locator)
    axes[i, 1].xaxis.set_major_formatter(date_formatter)
    axes[i, 1].tick_params(axis='x', rotation=45)  # 自动旋转X轴日期
    
    # 自动调整Y轴范围
    axes[i, 1].autoscale(enable=True, axis='y')

    # 绘制Simulated daily sterility
    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 2].plot(subset['DATE'], subset['Simulated daily sterility (optimum)'], label=variety, color=colors[j])
    axes[i, 2].set_title(f'{year} - Simulated daily sterility')
    axes[i, 2].set_xlabel('Date')
    axes[i, 2].set_ylabel('Simulated daily sterility')
    axes[i, 2].xaxis.set_major_locator(date_locator)
    axes[i, 2].xaxis.set_major_formatter(date_formatter)
    axes[i, 2].tick_params(axis='x', rotation=45)  # 自动旋转X轴日期
    
    # 自动调整Y轴范围
    axes[i, 2].autoscale(enable=True, axis='y')

    # 绘制Simulated cumulative sterility
    for j, variety in enumerate(year_df['Variety'].unique()):
        subset = year_df[year_df['Variety'] == variety]
        axes[i, 3].plot(subset['DATE'], subset['Simulated cumulative sterility (optimum)'], label=variety, color=colors[j])
    axes[i, 3].set_title(f'{year} - Simulated cumulative sterility')
    axes[i, 3].set_xlabel('Date')
    axes[i, 3].set_ylabel('Simulated cumulative sterility')
    axes[i, 3].xaxis.set_major_locator(date_locator)
    axes[i, 3].xaxis.set_major_formatter(date_formatter)
    axes[i, 3].tick_params(axis='x', rotation=45)  # 自动旋转X轴日期
    
    # 自动调整Y轴范围
    axes[i, 3].autoscale(enable=True, axis='y')

# 自动调整布局，确保标签不重叠
plt.tight_layout(h_pad=2.0, w_pad=1.0)

# 创建输出文件夹
output_path = r'C:\Users\sunti\Desktop\Fertility\Multi_year\En'
os.makedirs(output_path, exist_ok=True)

date_locator = mdates.DayLocator(interval=5)  # 每隔5天显示一次
date_formatter = mdates.DateFormatter('%m-%d')  # 设置日期格式

english_font = {'family': 'Times New Roman'}  # 英文字体

# 输出每个子图为单独的图片
for i, year in enumerate(unique_years):
    for j, title in enumerate(['Poisson probability', 'Cumulative probability', 'Simulated daily sterility (optimum)', 'Simulated cumulative sterility (optimum)']):
        # 创建单独的图像
        fig_single, ax_single = plt.subplots(figsize=(10, 10))
        
        # 绘制子图
        year_df = result_df[result_df['Year'] == year]
        for k, variety in enumerate(year_df['Variety'].unique()):
            subset = year_df[year_df['Variety'] == variety]
            ax_single.plot(subset['DATE'], subset[title], label=variety, color=colors[k])
        
        # 设置标题和标签
        ax_single.text(0.5, 0.95, f'{year}', transform=ax_single.transAxes, fontsize=28,
                       fontweight='bold', fontproperties=english_font, ha='center')
        ax_single.set_xlabel('Date', fontsize=32, fontweight='bold', fontproperties=english_font, labelpad=25)
        ax_single.set_ylabel(title, fontsize=32, fontweight='bold', fontproperties=english_font, labelpad=25)
        ax_single.xaxis.set_major_locator(date_locator)
        ax_single.xaxis.set_major_formatter(date_formatter)
        ax_single.tick_params(axis='x', rotation=45)  # 自动旋转X轴日期
        ax_single.tick_params(axis='both', labelcolor='black', width=2, length=12, direction='out')
        for label in ax_single.get_xticklabels() + ax_single.get_yticklabels():
            label.set_fontsize(24)
            label.set_fontweight('bold')
            label.set_family('Times New Roman')
        for spine in ax_single.spines.values():
            spine.set_linewidth(2)  # 设置边框线宽为2

        # 设置y轴范围和间隔
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

        # 保存子图，设置高DPI以提高清晰度
        output_file = os.path.join(output_path, f'{year}_{title}.png')
        fig_single.savefig(output_file, dpi=600, bbox_inches='tight')  # 设置dpi为600
        plt.close(fig_single)  # 关闭当前图形，释放内存
        print(f"Figures have saved to: {output_file}")

# 显示图像
plt.show()