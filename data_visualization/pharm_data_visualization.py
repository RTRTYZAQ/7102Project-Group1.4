import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import shutil

class PharmDataVisualization:
    def __init__(self, data_path="../data/Pharm Data_Data.csv", output_dir="./visualization_results"):
        """
        初始化药品数据可视化类
        
        参数:
        data_path: 药品数据路径
        output_dir: 输出目录
        """
        self.data = pd.read_csv(data_path)
        self.output_dir = output_dir
        
        # 预处理数据
        self.preprocess_data()
        
        # 获取唯一的国家和药品类别
        self.countries = self.data['Country'].unique()
        self.product_classes = self.data['Product Class'].unique()
        
        # 创建国家-药品类别组合
        self.country_product_pairs = []
        for country in self.countries:
            for product_class in self.product_classes:
                self.country_product_pairs.append((country, product_class))
        
        # # 确保输出目录存在
        # if os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
            
        print(f"Data loaded: {self.data.shape[0]} rows")
        print(f"Countries: {', '.join(self.countries)}")
        print(f"Product Classes: {', '.join(self.product_classes)}")
    
    def preprocess_data(self):
        """预处理数据"""
        # 创建国家-药品类别组合列
        self.data['Country-Product Class'] = self.data['Country'] + '-' + self.data['Product Class']
        
        # 处理日期
        self.data['Date'] = pd.to_datetime(self.data['Year'].astype(str) + '-' + 
                                         self.data['Month'].astype(str), format='%Y-%B')
        
        # 确保价格和销售额为数值
        self.data['Price'] = pd.to_numeric(self.data['Price'])
        self.data['Sales'] = pd.to_numeric(self.data['Sales'])
        self.data['Quantity'] = pd.to_numeric(self.data['Quantity'])
    
    def filter_data(self, country=None, product_class=None):
        """筛选特定国家和产品类别的数据"""
        filtered_data = self.data.copy()
        
        if country is not None:
            filtered_data = filtered_data[filtered_data['Country'] == country]
        
        if product_class is not None:
            filtered_data = filtered_data[filtered_data['Product Class'] == product_class]
            
        return filtered_data
    
    def create_output_directory(self, country=None, product_class=None):
        """创建输出目录"""
        if country is None and product_class is None:
            # 所有数据的可视化
            directory = os.path.join(self.output_dir, "all_data")
        elif country is not None and product_class is None:
            # 单一国家的可视化
            directory = os.path.join(self.output_dir, f"country_{country}")
        elif country is None and product_class is not None:
            # 单一药品类别的可视化
            directory = os.path.join(self.output_dir, f"class_{product_class}")
        else:
            # 国家-药品类别组合的可视化
            directory = os.path.join(self.output_dir, f"country_{country}_class_{product_class}")
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        return directory

    def visualize_price_summary(self, country=None, product_class=None):
        """可视化价格汇总信息"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 计算价格统计信息
        price_mean = data['Price'].mean()
        price_median = data['Price'].median()
        price_std = data['Price'].std()
        price_min = data['Price'].min()
        price_max = data['Price'].max()
        
        # 创建价格箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=data['Price'])
        
        plt.title(f"Price Distribution{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Price ($)")
        plt.grid(axis='x', alpha=0.3)
        
        # 添加统计信息
        plt.figtext(0.75, 0.75, f"Mean: ${price_mean:.2f}\nMedian: ${price_median:.2f}\nStd Dev: ${price_std:.2f}\nMin: ${price_min:.2f}\nMax: ${price_max:.2f}")
        
        # 保存图表
        output_path = os.path.join(directory, "price_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Price boxplot saved to {output_path}")
        
        # 创建价格直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(data['Price'], kde=True, bins=5)
        
        plt.title(f"Price Distribution Histogram{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Price ($)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "price_histogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Price histogram saved to {output_path}")
    
    def visualize_monthly_sales(self, country=None, product_class=None):
        """可视化月度销售额"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按月份分组并计算销售额
        monthly_sales = data.groupby('Date')['Sales'].sum().reset_index()
        monthly_sales = monthly_sales.sort_values('Date')
        
        # 创建月度销售额折线图
        plt.figure(figsize=(12, 6))
        
        # 绘制折线图
        plt.plot(monthly_sales['Date'], monthly_sales['Sales'], marker='o', linestyle='-', color='#1f77b4')
        
        plt.title(f"Monthly Sales Trend{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Date")
        plt.ylabel("Sales ($)")
        plt.grid(True, alpha=0.3)
        
        # 设置日期格式化
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        
        # 保存图表
        output_path = os.path.join(directory, "monthly_sales_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Monthly sales trend visualization saved to {output_path}")
        
        # 创建月度销售额柱状图
        plt.figure(figsize=(12, 6))
        
        # 绘制柱状图
        bars = plt.bar(monthly_sales['Date'], monthly_sales['Sales'], width=20, alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            # 仅给最大值添加标签
            if bar.get_height() == monthly_sales['Sales'].max():
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height,
                         s = f'${height:,.0f}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title(f"Monthly Sales{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Date")
        plt.ylabel("Sales ($)")
        plt.grid(axis='y', alpha=0.3)
        
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # 保存图表
        output_path = os.path.join(directory, "monthly_sales_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Monthly sales bar chart saved to {output_path}")

    def visualize_monthly_quantity(self, country=None, product_class=None):
        """可视化月度销量"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按月份分组并计算销量
        monthly_quantity = data.groupby('Date')['Quantity'].sum().reset_index()
        monthly_quantity = monthly_quantity.sort_values('Date')
        
        # 创建月度销量折线图
        plt.figure(figsize=(12, 6))
        
        # 绘制折线图
        plt.plot(monthly_quantity['Date'], monthly_quantity['Quantity'], marker='o', linestyle='-', color='#1f77b4')
        
        plt.title(f"Monthly Quantity Trend{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.grid(True, alpha=0.3)
        
        # 设置日期格式化
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        
        # 保存图表
        output_path = os.path.join(directory, "monthly_quantity_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Monthly quantity trend visualization saved to {output_path}")
        
        # 创建月度销量柱状图
        plt.figure(figsize=(12, 6))
        
        # 绘制柱状图
        bars = plt.bar(monthly_quantity['Date'], monthly_quantity['Quantity'], width=20, alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            # 仅给最大值添加标签
            if bar.get_height() == monthly_quantity['Quantity'].max():
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height,
                        s = f'{height:,.0f}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title(f"Monthly Quantity{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.grid(axis='y', alpha=0.3)
        
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # 保存图表
        output_path = os.path.join(directory, "monthly_quantity_bar.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Monthly quantity bar chart saved to {output_path}")

    def visualize_top_products(self, country=None, product_class=None, top_n=10):
        """可视化畅销产品"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按产品分组并计算销售额
        product_sales = data.groupby('Product Name')['Sales'].sum().reset_index()
        product_sales = product_sales.sort_values('Sales', ascending=False).head(top_n)
        
        # 创建畅销产品条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(product_sales['Product Name'], product_sales['Sales'], color='skyblue')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'${width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Products by Sales{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Sales ($)")
        plt.ylabel("Product Name")
        plt.grid(axis='x', alpha=0.3)
        
        plt.gca().invert_yaxis()  # 让销售额最高的产品显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_products_by_sales.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top products visualization saved to {output_path}")
        
        # 按产品分组并计算销售数量
        product_quantity = data.groupby('Product Name')['Quantity'].sum().reset_index()
        product_quantity = product_quantity.sort_values('Quantity', ascending=False).head(top_n)
        
        # 创建销量最高产品条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(product_quantity['Product Name'], product_quantity['Quantity'], color='lightgreen')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Products by Quantity{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Quantity Sold")
        plt.ylabel("Product Name")
        plt.grid(axis='x', alpha=0.3)
        
        plt.gca().invert_yaxis()  # 让销量最高的产品显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_products_by_quantity.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top products by quantity visualization saved to {output_path}")

    def visualize_channel_sales_distribution(self, country=None, product_class=None):
        """可视化销售渠道分布"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按渠道分组并计算销售额
        channel_sales = data.groupby('Channel')['Sales'].sum().reset_index()

        # 创建渠道分布饼图
        plt.figure(figsize=(10, 8))

        # 计算百分比
        total_sales = channel_sales['Sales'].sum()
        percentages = [(sales / total_sales) * 100 for sales in channel_sales['Sales']]

        # 创建自定义标签函数，显示百分比和销售额
        def my_autopct(pct):
            # 根据百分比找到对应的销售额
            idx = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = channel_sales['Sales'].iloc[idx]
            # 返回两行格式：百分比在上，销售额在下
            return f"{pct:.1f}%\n${val:,.0f}"

        # 绘制饼图 - 使用自定义标签函数
        plt.pie(channel_sales['Sales'], labels=channel_sales['Channel'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(channel_sales),
                textprops={'fontsize': 12})

        plt.axis('equal')  # 确保饼图是圆的
        plt.title(f"Sales Distribution by Channel{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")

        # 保存图表
        output_path = os.path.join(directory, "channel_distribution_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Channel distribution pie chart saved to {output_path}")
        

        # 按渠道和子渠道分组
        subchannel_sales = data.groupby(['Channel', 'Sub-channel'])['Sales'].sum().reset_index()
        
        # 用于颜色映射
        channel_colors = {
            'Hospital': 'skyblue',
            'Pharmacy': 'lightgreen'
        }
        
        # 创建子渠道条形图
        plt.figure(figsize=(12, 8))
        
        # 获取唯一的渠道
        channels = subchannel_sales['Channel'].unique()
        
        # 为每个渠道创建子图
        for i, channel in enumerate(channels):
            channel_data = subchannel_sales[subchannel_sales['Channel'] == channel]
            channel_data = channel_data.sort_values('Sales', ascending=False)
            
            plt.subplot(len(channels), 1, i+1)
            bars = plt.barh(channel_data['Sub-channel'], channel_data['Sales'], 
                           color=channel_colors.get(channel, 'gray'))
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                         f'${width:,.0f}', va='center', fontsize=9)
            
            plt.title(f"{channel} - Sales by Sub-channel")
            plt.xlabel("Sales ($)")
            plt.grid(axis='x', alpha=0.3)
            
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Sales by Channel and Sub-channel{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}", 
                    fontsize=14, y=0.98)
        
        # 保存图表
        output_path = os.path.join(directory, "subchannel_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sub-channel distribution chart saved to {output_path}")

    def visualize_channel_quantity_distribution(self, country=None, product_class=None):
        """可视化销售渠道销量分布"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按渠道分组并计算销量
        channel_quantity = data.groupby('Channel')['Quantity'].sum().reset_index()
        # 创建渠道分布饼图
        plt.figure(figsize=(10, 8))

        # 计算百分比
        total_quantity = channel_quantity['Quantity'].sum()
        percentages = [(quantity / total_quantity) * 100 for quantity in channel_quantity['Quantity']]

        # 创建自定义标签函数，显示百分比和销量
        def my_autopct(pct):
            # 根据百分比找到对应的销量
            idx = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = channel_quantity['Quantity'].iloc[idx]
            # 返回两行格式：百分比在上，销量在下
            return f"{pct:.1f}%\n{val:.0f}"

        # 绘制饼图 - 使用自定义标签函数
        plt.pie(channel_quantity['Quantity'], labels=channel_quantity['Channel'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(channel_quantity),
                textprops={'fontsize': 12})

        plt.axis('equal')  # 确保饼图是圆的
        plt.title(f"Quantity Distribution by Channel{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")

        # 保存图表
        output_path = os.path.join(directory, "channel_quantity_distribution_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Channel quantity distribution pie chart saved to {output_path}")
        

        # 按渠道和子渠道分组
        subchannel_quantity = data.groupby(['Channel', 'Sub-channel'])['Quantity'].sum().reset_index()
        
        # 用于颜色映射
        channel_colors = {
            'Hospital': 'skyblue',
            'Pharmacy': 'lightgreen'
        }
        
        # 创建子渠道条形图
        plt.figure(figsize=(12, 8))
        
        # 获取唯一的渠道
        channels = subchannel_quantity['Channel'].unique()
        
        # 为每个渠道创建子图
        for i, channel in enumerate(channels):
            channel_data = subchannel_quantity[subchannel_quantity['Channel'] == channel]
            channel_data = channel_data.sort_values('Quantity', ascending=False)
            
            plt.subplot(len(channels), 1, i+1)
            bars = plt.barh(channel_data['Sub-channel'], channel_data['Quantity'], 
                        color=channel_colors.get(channel, 'gray'))
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:,.0f}', va='center', fontsize=9)
            
            plt.title(f"{channel} - Quantity by Sub-channel")
            plt.xlabel("Quantity")
            plt.grid(axis='x', alpha=0.3)
            
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f"Quantity by Channel and Sub-channel{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}", 
                    fontsize=14, y=0.98)
        
        # 保存图表
        output_path = os.path.join(directory, "subchannel_quantity_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sub-channel quantity distribution chart saved to {output_path}")

    def visualize_sales_by_distributor(self, country=None, product_class=None, top_n=10):
        """可视化各经销商的销售额"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按经销商分组并计算销售额
        distributor_sales = data.groupby('Distributor')['Sales'].sum().reset_index()
        distributor_sales = distributor_sales.sort_values('Sales', ascending=False).head(top_n)
        
        # 创建经销商销售额条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(distributor_sales['Distributor'], distributor_sales['Sales'], color='lightcoral')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'${width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Distributors by Sales{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Sales ($)")
        plt.ylabel("Distributor")
        plt.grid(axis='x', alpha=0.3)
        
        plt.gca().invert_yaxis()  # 让销售额最高的经销商显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_distributors.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top distributors visualization saved to {output_path}")
        
        # 按经销商分组并计算订单数量
        distributor_orders = data.groupby('Distributor').size().reset_index(name='Orders')
        distributor_orders = distributor_orders.sort_values('Orders', ascending=False).head(top_n)
        
        # 创建经销商订单数量条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(distributor_orders['Distributor'], distributor_orders['Orders'], color='mediumpurple')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Distributors by Order Count{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Number of Orders")
        plt.ylabel("Distributor")
        plt.grid(axis='x', alpha=0.3)
        
        plt.gca().invert_yaxis()  # 让订单数量最高的经销商显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_distributors_by_orders.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top distributors by orders visualization saved to {output_path}")
    
    def visualize_sales_reps_performance(self, country=None, product_class=None, top_n=10):
        """可视化销售代表业绩"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按销售代表分组并计算销售额
        rep_sales = data.groupby('Name of Sales Rep')['Sales'].sum().reset_index()
        rep_sales = rep_sales.sort_values('Sales', ascending=False).head(top_n)
        
        # 创建销售代表业绩条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(rep_sales['Name of Sales Rep'], rep_sales['Sales'], color='orange')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'${width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Sales Representatives by Sales{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Sales ($)")
        plt.ylabel("Sales Representative")
        plt.grid(axis='x', alpha=0.3)
        
        plt.gca().invert_yaxis()  # 让销售额最高的代表显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_sales_reps.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top sales representatives visualization saved to {output_path}")
        
    def visualize_geo_distribution(self, country=None, product_class=None, top_n=10):
        """可视化地理分布"""
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
            
        directory = self.create_output_directory(country, product_class)
        
        # 按城市分组并计算销售额
        city_sales = data.groupby('City')['Sales'].sum().reset_index()
        city_sales = city_sales.sort_values('Sales', ascending=False).head(top_n)
        
        # 创建城市销售额条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(city_sales['City'], city_sales['Sales'], color='teal')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'${width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Cities by Sales{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
        plt.xlabel("Sales ($)")
        plt.ylabel("City")
        plt.grid(axis='x', alpha=0.3)
        
        plt.gca().invert_yaxis()  # 让销售额最高的城市显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_cities.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top cities visualization saved to {output_path}")
        
        # 尝试创建地图可视化（如果有经纬度信息）
        if 'Latitude' in data.columns and 'Longitude' in data.columns:
            try:
                import cartopy.crs as ccrs
                import cartopy.feature as cfeature
                
                # 准备地图数据
                city_geo = data.groupby(['City', 'Latitude', 'Longitude'])['Sales'].sum().reset_index()
                city_geo = city_geo.sort_values('Sales', ascending=False)
                
                # 创建地图
                plt.figure(figsize=(15, 10))
                ax = plt.axes(projection=ccrs.PlateCarree())
                
                # 添加地图特征
                ax.add_feature(cfeature.LAND)
                ax.add_feature(cfeature.OCEAN)
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                
                # 设置地图范围
                if country:
                    # 根据国家设置地图范围
                    min_lat = city_geo['Latitude'].min() - 1
                    max_lat = city_geo['Latitude'].max() + 1
                    min_lon = city_geo['Longitude'].min() - 1
                    max_lon = city_geo['Longitude'].max() + 1
                    ax.set_extent([min_lon, max_lon, min_lat, max_lat])
                
                # 按销售额大小绘制散点
                sizes = city_geo['Sales'] / city_geo['Sales'].max() * 500
                sc = ax.scatter(city_geo['Longitude'], city_geo['Latitude'], 
                              s=sizes, c=city_geo['Sales'], cmap='viridis', 
                              alpha=0.7, transform=ccrs.PlateCarree())
                
                # 添加颜色条
                plt.colorbar(sc, label='Sales ($)', shrink=0.6)
                
                # 为前几个城市添加标签
                for i, row in city_geo.head(5).iterrows():
                    ax.text(row['Longitude'] + 0.1, row['Latitude'] + 0.1, row['City'], 
                          transform=ccrs.PlateCarree(), fontsize=9)
                
                plt.title(f"Geographic Distribution of Sales{' - ' + country if country else ''}{' - ' + product_class if product_class else ''}")
                
                # 保存地图
                output_path = os.path.join(directory, "sales_geo_map.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Geographic map saved to {output_path}")
            except Exception as e:
                print(f"Could not create map visualization: {e}")
                print("Consider installing cartopy with: pip install cartopy")
     
    def visualize_country_sales_comparison(self, product_class=None):
        """比较不同国家的销售情况"""
        if len(self.countries) <= 1:
            print("Need at least 2 countries for comparison")
            return
            
        directory = self.create_output_directory(product_class=product_class)
        
        # 按国家分组并计算销售额
        country_sales = self.data.groupby('Country')['Sales'].sum().reset_index()
        country_sales = country_sales.sort_values('Sales', ascending=False)
        
        # 创建国家销售额饼图
        plt.figure(figsize=(10, 8))

        # 计算百分比
        total_sales = country_sales['Sales'].sum()
        percentages = [(sales / total_sales) * 100 for sales in country_sales['Sales']]

        # 创建自定义标签函数，显示百分比和销售额
        def my_autopct(pct):
            # 根据百分比找到对应的销售额
            idx = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = country_sales['Sales'].iloc[idx]
            # 返回两行格式：百分比在上，销售额在下
            return f"{pct:.1f}%\n${val:,.0f}"

        # 绘制饼图 - 使用自定义标签函数
        plt.pie(country_sales['Sales'], labels=country_sales['Country'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(country_sales),
                textprops={'fontsize': 12})

        plt.axis('equal')  # 确保饼图是圆的
        plt.title(f"Sales Distribution by Country{' - ' + product_class if product_class else ''}")

        # 保存图表
        output_path = os.path.join(directory, "country_sales_comparison_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Country sales comparison pie chart saved to {output_path}")
        
        # 创建各国家药品类别分布图
        if len(self.product_classes) > 1:
            # 按国家和产品类别分组并计算销售额
            country_class_dist = self.data.groupby(['Country', 'Product Class'])['Sales'].sum().reset_index()
            
            # 创建多子图
            fig, axes = plt.subplots(len(self.countries), 1, figsize=(12, 5 * len(self.countries)))
            
            if len(self.countries) == 1:
                axes = [axes]
                
            for i, country in enumerate(self.countries):
                country_data = country_class_dist[country_class_dist['Country'] == country]
                country_data = country_data.sort_values('Sales', ascending=False)
                
                # 绘制条形图
                ax = axes[i]
                bars = ax.bar(country_data['Product Class'], country_data['Sales'], color=f'C{i}')
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                           f'${height:,.0f}', ha='center', va='bottom', rotation=0, fontsize=9)
                
                ax.set_title(f"{country} - Sales by Product Class")
                ax.set_xlabel("Product Class")
                ax.set_ylabel("Sales ($)")
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(directory, "country_product_class_distribution.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Country product class distribution visualization saved to {output_path}")
            
    def visualize_channel_sales_comparison(self, country=None, product_class=None):
        """比较不同销售渠道的销售情况"""
        # 筛选数据
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
        
        # 检查渠道数量
        channels = data['Channel'].unique()
        if len(channels) <= 1:
            print("Need at least 2 channels for comparison")
            return
            
        directory = self.create_output_directory(country=country, product_class=product_class)
        
        # 按渠道分组并计算销售额
        channel_sales = data.groupby('Channel')['Sales'].sum().reset_index()
        channel_sales = channel_sales.sort_values('Sales', ascending=False)
        
        
        # 创建各渠道药品类别分布图
        if len(self.product_classes) > 1:
            # 按渠道和产品类别分组并计算销售额
            channel_class_dist = data.groupby(['Channel', 'Product Class'])['Sales'].sum().reset_index()
            
            # 创建多子图
            fig, axes = plt.subplots(len(channels), 1, figsize=(12, 5 * len(channels)))
            
            if len(channels) == 1:
                axes = [axes]
                
            for i, channel in enumerate(channels):
                channel_data = channel_class_dist[channel_class_dist['Channel'] == channel]
                channel_data = channel_data.sort_values('Sales', ascending=False)
                
                # 绘制条形图
                ax = axes[i]
                bars = ax.bar(channel_data['Product Class'], channel_data['Sales'], color=f'C{i}')
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'${height:,.0f}', ha='center', va='bottom', rotation=0, fontsize=9)
                
                ax.set_title(f"{channel} - Sales by Product Class")
                ax.set_xlabel("Product Class")
                ax.set_ylabel("Sales ($)")
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            title = "Sales by Channel and Product Class"
            if country:
                title += f" - {country}"
            
            plt.suptitle(title, fontsize=14, y=0.98)
            plt.subplots_adjust(top=0.9)
            
            # 保存图表
            output_path = os.path.join(directory, "channel_product_class_distribution.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Channel product class distribution visualization saved to {output_path}")

    def visualize_product_class_sales_comparison(self, country=None):
        """比较不同产品类别的销售情况"""
        if len(self.product_classes) <= 1:
            print("Need at least 2 product classes for comparison")
            return
            
        directory = self.create_output_directory(country=country)
        
        # 筛选数据
        if country:
            data = self.data[self.data['Country'] == country]
        else:
            data = self.data
        
        # 按产品类别分组并计算销售额
        class_sales = data.groupby('Product Class')['Sales'].sum().reset_index()
        class_sales = class_sales.sort_values('Sales', ascending=False)
        
        # 创建产品类别销售额条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(class_sales['Product Class'], class_sales['Sales'], color='darkviolet')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'${width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Sales by Product Class{' - ' + country if country else ''}")
        plt.xlabel("Sales ($)")
        plt.ylabel("Product Class")
        plt.grid(axis='x', alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_sales.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class sales visualization saved to {output_path}")
        
        # 创建产品类别销售占比饼图
        plt.figure(figsize=(10, 8))
        
        # 计算百分比
        total_sales = class_sales['Sales'].sum()
        percentages = [(sales / total_sales) * 100 for sales in class_sales['Sales']]
        
        def my_autopct(pct):
            # 根据百分比找到对应的销售额
            idx = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = class_sales['Sales'].iloc[idx]
            return f"{pct:.1f}%\n${val:,.0f}"
        
        # 绘制饼图
        plt.pie(class_sales['Sales'], labels=class_sales['Product Class'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(class_sales),
                textprops={'fontsize': 12})
        
        plt.axis('equal')  # 确保饼图是圆的
        plt.title(f"Product Class Sales Distribution{' - ' + country if country else ''}")
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_distribution_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class distribution pie chart saved to {output_path}")
        
        # 创建产品类别平均价格条形图
        class_avg_price = data.groupby('Product Class')['Price'].mean().reset_index()
        class_avg_price = class_avg_price.sort_values('Price', ascending=False)
        
        plt.figure(figsize=(10, 6))
        
        # 绘制条形图
        bars = plt.bar(class_avg_price['Product Class'], class_avg_price['Price'], color='crimson')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'${height:.2f}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title(f"Average Price by Product Class{' - ' + country if country else ''}")
        plt.xlabel("Product Class")
        plt.ylabel("Average Price ($)")
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_avg_price.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class average price visualization saved to {output_path}")

    def visualize_country_quantity_comparison(self, product_class=None):
        """比较不同国家的销量情况"""
        if len(self.countries) <= 1:
            print("Need at least 2 countries for comparison")
            return
            
        directory = self.create_output_directory(product_class=product_class)
        
        # 按国家分组并计算销量
        country_quantity = self.data.groupby('Country')['Quantity'].sum().reset_index()
        country_quantity = country_quantity.sort_values('Quantity', ascending=False)
        
        # 创建国家销量饼图
        plt.figure(figsize=(10, 8))

        # 计算百分比
        total_quantity = country_quantity['Quantity'].sum()
        percentages = [(quantity / total_quantity) * 100 for quantity in country_quantity['Quantity']]

        # 创建自定义标签函数，显示百分比和销量
        def my_autopct(pct):
            # 根据百分比找到对应的销量
            idx = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = country_quantity['Quantity'].iloc[idx]
            # 返回两行格式：百分比在上，销量在下
            return f"{pct:.1f}%\n{val:,.0f}"

        # 绘制饼图 - 使用自定义标签函数
        plt.pie(country_quantity['Quantity'], labels=country_quantity['Country'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(country_quantity),
                textprops={'fontsize': 12})

        plt.axis('equal')  # 确保饼图是圆的
        plt.title(f"Quantity Distribution by Country{' - ' + product_class if product_class else ''}")

        # 保存图表
        output_path = os.path.join(directory, "country_quantity_comparison_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Country quantity comparison pie chart saved to {output_path}")
        
        # 创建各国家药品类别销量分布图
        if len(self.product_classes) > 1:
            # 按国家和产品类别分组并计算销量
            country_class_dist = self.data.groupby(['Country', 'Product Class'])['Quantity'].sum().reset_index()
            
            # 创建多子图
            fig, axes = plt.subplots(len(self.countries), 1, figsize=(12, 5 * len(self.countries)))
            
            if len(self.countries) == 1:
                axes = [axes]
                
            for i, country in enumerate(self.countries):
                country_data = country_class_dist[country_class_dist['Country'] == country]
                country_data = country_data.sort_values('Quantity', ascending=False)
                
                # 绘制条形图
                ax = axes[i]
                bars = ax.bar(country_data['Product Class'], country_data['Quantity'], color=f'C{i}')
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{height:,}', ha='center', va='bottom', rotation=0, fontsize=9)
                
                ax.set_title(f"{country} - Quantity by Product Class")
                ax.set_xlabel("Product Class")
                ax.set_ylabel("Quantity")
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            output_path = os.path.join(directory, "country_product_class_quantity_distribution.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Country product class quantity distribution visualization saved to {output_path}")

    def visualize_channel_quantity_comparison(self, country=None, product_class=None):
        """比较不同销售渠道的销量情况"""
        # 筛选数据
        data = self.filter_data(country, product_class)
        if data.empty:
            print(f"No data available for country={country}, product_class={product_class}")
            return
        
        # 检查渠道数量
        channels = data['Channel'].unique()
        if len(channels) <= 1:
            print("Need at least 2 channels for comparison")
            return
            
        directory = self.create_output_directory(country=country, product_class=product_class)
        
        # 按渠道分组并计算销量
        channel_quantity = data.groupby('Channel')['Quantity'].sum().reset_index()
        channel_quantity = channel_quantity.sort_values('Quantity', ascending=False)
        
        # 创建渠道销量饼图
        plt.figure(figsize=(10, 8))

        # 计算百分比
        total_quantity = channel_quantity['Quantity'].sum()
        percentages = [(quantity / total_quantity) * 100 for quantity in channel_quantity['Quantity']]

        # 创建自定义标签函数，显示百分比和销量
        def my_autopct(pct):
            # 根据百分比找到对应的销量
            idx = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = channel_quantity['Quantity'].iloc[idx]
            # 返回两行格式：百分比在上，销量在下
            return f"{pct:.1f}%\n{val:0f}"

        # 绘制饼图 - 使用自定义标签函数
        plt.pie(channel_quantity['Quantity'], labels=channel_quantity['Channel'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(channel_quantity),
                textprops={'fontsize': 12})

        plt.axis('equal')  # 确保饼图是圆的
        title_parts = []
        if country:
            title_parts.append(country)
        if product_class:
            title_parts.append(product_class)
        
        title = "Quantity Distribution by Channel"
        if title_parts:
            title += " - " + " / ".join(title_parts)
        
        plt.title(title)
        
        # 保存图表
        output_path = os.path.join(directory, "channel_quantity_comparison_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Channel quantity comparison pie chart saved to {output_path}")
        
        # 创建各渠道药品类别销量分布图
        if len(self.product_classes) > 1:
            # 按渠道和产品类别分组并计算销量
            channel_class_dist = data.groupby(['Channel', 'Product Class'])['Quantity'].sum().reset_index()
            
            # 创建多子图
            fig, axes = plt.subplots(len(channels), 1, figsize=(12, 5 * len(channels)))
            
            if len(channels) == 1:
                axes = [axes]
                
            for i, channel in enumerate(channels):
                channel_data = channel_class_dist[channel_class_dist['Channel'] == channel]
                channel_data = channel_data.sort_values('Quantity', ascending=False)
                
                # 绘制条形图
                ax = axes[i]
                bars = ax.bar(channel_data['Product Class'], channel_data['Quantity'], color=f'C{i}')
                
                # 添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                        f'{height:,}', ha='center', va='bottom', rotation=0, fontsize=9)
                
                ax.set_title(f"{channel} - Quantity by Product Class")
                ax.set_xlabel("Product Class")
                ax.set_ylabel("Quantity")
                ax.grid(axis='y', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            title = "Quantity by Channel and Product Class"
            if country:
                title += f" - {country}"
            
            plt.suptitle(title, fontsize=14, y=0.98)
            plt.subplots_adjust(top=0.9)
            
            # 保存图表
            output_path = os.path.join(directory, "channel_product_class_quantity_distribution.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Channel product class quantity distribution visualization saved to {output_path}")

    def visualize_product_class_quantity_comparison(self, country=None):
        """比较不同产品类别的销量情况"""
        if len(self.product_classes) <= 1:
            print("Need at least 2 product classes for comparison")
            return
            
        directory = self.create_output_directory(country=country)
        
        # 筛选数据
        if country:
            data = self.data[self.data['Country'] == country]
        else:
            data = self.data
        
        # 按产品类别分组并计算销量
        class_quantity = data.groupby('Product Class')['Quantity'].sum().reset_index()
        class_quantity = class_quantity.sort_values('Quantity', ascending=False)
        
        # 创建产品类别销量条形图
        plt.figure(figsize=(12, 8))
        
        # 绘制水平条形图
        bars = plt.barh(class_quantity['Product Class'], class_quantity['Quantity'], color='darkviolet')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Quantity by Product Class{' - ' + country if country else ''}")
        plt.xlabel("Quantity")
        plt.ylabel("Product Class")
        plt.grid(axis='x', alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_quantity.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class quantity visualization saved to {output_path}")
        
        # 创建产品类别销量占比饼图
        plt.figure(figsize=(10, 8))
        
        # 计算百分比
        total_quantity = class_quantity['Quantity'].sum()
        percentages = [(quantity / total_quantity) * 100 for quantity in class_quantity['Quantity']]
        
        # 创建自定义标签函数，显示百分比和销量
        def my_autopct(pct):
            # 根据百分比找到对应的销量
            idx = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = class_quantity['Quantity'].iloc[idx]
            # 返回两行格式：百分比在上，销量在下
            return f"{pct:.1f}%\n{val:,.0f}"
        
        # 绘制饼图 - 使用自定义标签函数
        plt.pie(class_quantity['Quantity'], labels=class_quantity['Product Class'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(class_quantity),
                textprops={'fontsize': 12})
        
        plt.axis('equal')  # 确保饼图是圆的
        plt.title(f"Product Class Quantity Distribution{' - ' + country if country else ''}")
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_quantity_distribution_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class quantity distribution pie chart saved to {output_path}")
        
        # 创建产品类别平均销量条形图
        class_avg_quantity = data.groupby('Product Class')['Quantity'].mean().reset_index()
        class_avg_quantity = class_avg_quantity.sort_values('Quantity', ascending=False)
        
        plt.figure(figsize=(10, 6))
        
        # 绘制条形图
        bars = plt.bar(class_avg_quantity['Product Class'], class_avg_quantity['Quantity'], color='crimson')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{height:.1f}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title(f"Average Quantity per Order by Product Class{' - ' + country if country else ''}")
        plt.xlabel("Product Class")
        plt.ylabel("Average Quantity")
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_avg_quantity.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class average quantity visualization saved to {output_path}")

    def run_single_visualization(self, country=None, product_class=None):
        """执行单个 <国家-药品类别> 可视化"""
        print(f"\nGenerating visualizations for country: {country}, product class: {product_class}...")
        self.visualize_price_summary(country=country, product_class=product_class)
        self.visualize_monthly_sales(country=country, product_class=product_class)
        self.visualize_monthly_quantity(country=country, product_class=product_class)
        self.visualize_top_products(country=country, product_class=product_class)
        self.visualize_channel_sales_distribution(country=country, product_class=product_class)
        self.visualize_channel_quantity_distribution(country=country, product_class=product_class)
        self.visualize_sales_by_distributor(country=country, product_class=product_class)
        self.visualize_sales_reps_performance(country=country, product_class=product_class)
        self.visualize_geo_distribution(country=country, product_class=product_class)

    def run_all_visualizations(self):
        """执行总体数据可视化"""
        print("Starting all visualizations...")
        
        # 全局可视化
        print("\nGenerating global visualizations...")
        self.visualize_price_summary()
        self.visualize_monthly_sales()
        self.visualize_monthly_quantity()
        self.visualize_top_products()
        self.visualize_channel_sales_distribution()
        self.visualize_channel_quantity_distribution()
        self.visualize_sales_by_distributor()
        self.visualize_sales_reps_performance()
        self.visualize_geo_distribution()
        self.visualize_country_sales_comparison()
        self.visualize_channel_sales_comparison()
        self.visualize_product_class_sales_comparison()
        self.visualize_country_quantity_comparison()
        self.visualize_channel_quantity_comparison()
        self.visualize_product_class_quantity_comparison()

        print("\nAll visualizations completed!")
    
    def run_all_single_visualizations(self):
        """执行所有 <国家-药品类别> 可视化"""
        print("Starting all single visualizations...")
        
        for country in self.countries:
            for product_class in self.product_classes:
                self.run_single_visualization(country=country, product_class=product_class)
        print("\nAll single visualizations completed!")
