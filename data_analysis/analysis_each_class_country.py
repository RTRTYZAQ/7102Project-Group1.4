#%%
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 数据加载
data = pd.read_csv('./data/Pharm Data_Data.csv')
data['Product Class-Country'] = data['Product Class'] + '-' + data['Country']

# 定义分析类
class Analysis:
    def __init__(self, data):
        self.data = data

    def analyze_sales(self):
        sales_summary = self.data.groupby('Product Class-Country')['Sales'].sum().to_dict()
        return {'sales_summary': sales_summary}

    def analyze_quantity(self):
        quantity_summary = self.data.groupby('Product Class-Country')['Quantity'].sum().to_dict()
        return {'quantity_summary': quantity_summary}

    def analyze_price(self):
        avg_price = self.data.groupby('Product Class-Country')['Price'].mean().round(2).to_dict()
        return {'avg_price': avg_price}

    def analyze_all(self):
        return {
            **self.analyze_sales(),
            **self.analyze_quantity(),
            **self.analyze_price()
        }

# 定义可视化类
class Visualization:
    def __init__(self, data, analysis_result, category):
        self.data = data
        self.analysis_result = analysis_result
        self.category = category

    def plot_sales(self):
        sales_summary = self.analysis_result['sales_summary']
        plt.figure(figsize=(10, 6))
        plt.bar(sales_summary.keys(), sales_summary.values())
        plt.xlabel('Product Class-Country')
        plt.ylabel('Total Sales')
        plt.title(f'Sales Distribution for {self.category}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'./analysis_result_{self.category}/sales_distribution_{self.category}.png')

    def plot_quantity(self):
        quantity_summary = self.analysis_result['quantity_summary']
        plt.figure(figsize=(10, 6))
        plt.bar(quantity_summary.keys(), quantity_summary.values(), color='orange')
        plt.xlabel('Product Class-Country')
        plt.ylabel('Total Quantity')
        plt.title(f'Quantity Distribution for {self.category}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'./analysis_result_{self.category}/quantity_distribution_{self.category}.png')

    def plot_avg_price(self):
        avg_price = self.analysis_result['avg_price']
        plt.figure(figsize=(10, 6))
        plt.bar(avg_price.keys(), avg_price.values(), color='green')
        plt.xlabel('Product Class-Country')
        plt.ylabel('Average Price')
        plt.title(f'Average Price for {self.category}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'./analysis_result_{self.category}/avg_price_{self.category}.png')

# 主循环
for category in data['Product Class-Country'].unique():
    print(f'Processing: {category}')
    if not os.path.exists(f'./analysis_result_{category}'):
        os.makedirs(f'./analysis_result_{category}')
    category_data = data[data['Product Class-Country'] == category]
    analysis = Analysis(category_data)
    analysis_result = analysis.analyze_all()
    with open(f'./analysis_result_{category}/analysis_result_{category}.json', 'w') as f:
        json.dump(analysis_result, f, indent=4)
    visualization = Visualization(category_data, analysis_result, category)
    visualization.plot_sales()
    visualization.plot_quantity()
    visualization.plot_avg_price()