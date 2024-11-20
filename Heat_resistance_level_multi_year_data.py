import pandas as pd

# 读取Excel文件
file_2022_2023 = r'C:\Users\sunti\Desktop\Fertility\Multi_year\En\2022_2023_Flowering_distribution_Sterility.xlsx'
df_2022_2023 = pd.read_excel(file_2022_2023)

# 提取品种、温度阈值和不育率（删除重复项）
df_2022_2023_filtered = df_2022_2023[['Variety', 'Temperature threshold', 'Observed sterility']].drop_duplicates()

# 删除缺失值的记录，保证参与分级的数据没有缺失（针对不育率观测值和温度阈值）
df_2022_2023_filtered = df_2022_2023_filtered.dropna(subset=['Observed sterility', 'Temperature threshold'])

# 查找每个品种在2022年和2023年都有不缺失的‘Observed sterility’数据
valid_varieties = df_2022_2023_filtered.groupby('Variety').filter(lambda x: x['Observed sterility'].notna().sum() == 2)['Variety'].unique()

# 过滤出在两年都有记录的品种
valid_data = df_2022_2023_filtered[df_2022_2023_filtered['Variety'].isin(valid_varieties)]

# 保留每个品种只保留一条记录（这里选择保留第一条记录，您可以根据需要调整）
valid_data = valid_data.drop_duplicates(subset='Variety', keep='first')

# 使用 pd.qcut 进行分级（按分位数分级，分成5个等级）
valid_data['Heat_resistance_level'] = pd.qcut(valid_data['Temperature threshold'], q=5, labels=[1, 2, 3, 4, 5])

# 输出结果
final_df = valid_data[['Variety', 'Temperature threshold', 'Heat_resistance_level']].sort_values(by='Heat_resistance_level', ascending=False)

# 输出到新的Excel文件
output_file = r'C:\Users\sunti\Desktop\Fertility\Multi_year\En\Heat_resistance_level_multi_year_data.xlsx'
final_df.to_excel(output_file, index=False)

print(f"结果已输出到：{output_file}")
