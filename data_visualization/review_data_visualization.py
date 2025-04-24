import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import shutil
from wordcloud import WordCloud
import re

class ReviewDataVisualization:
    def __init__(self, data_path, output_dir="./review_visualization_results"):
        """
        初始化评论数据可视化类
        
        参数:
        data_path: 评论数据路径
        output_dir: 输出目录
        """
        self.data_path = data_path
        self.output_dir = output_dir
        
        # 确保输出目录存在
        # if os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理数据
        self.process_data()
        self.load_data()
    
    def process_data(self):
        """预处理评论数据"""
        # print(f"Processing data: {self.data_path}")
        pass
    
    def load_data(self):
        """加载评论数据"""
        self.data = pd.read_csv(self.data_path)
        
        # 转换日期为datetime格式
        try:
            self.data['date'] = pd.to_datetime(self.data['date'], format='%d-%b-%y')
        except:
            self.data['date'] = pd.to_datetime(self.data['date'])
        
        # 计算评论长度
        self.data['review_length'] = self.data['review'].apply(lambda x: len(str(x)))
        
        # 确保评分和有用计数为数值型
        self.data['rating'] = pd.to_numeric(self.data['rating'])
        self.data['usefulCount'] = pd.to_numeric(self.data['usefulCount'])
        
        # 获取唯一的药品和病症
        self.drugs = self.data['drugName'].unique()
        self.conditions = self.data['condition'].unique()
        
        # 如果存在产品类别，获取唯一类别
        if 'Product Class' in self.data.columns:
            self.product_classes = self.data[self.data['Product Class'] != 'Else']['Product Class'].unique()
        else:
            self.product_classes = []
            
        print(f"Data loaded: {self.data.shape[0]} reviews")
        print(f"Number of drugs: {len(self.drugs)}")
        print(f"Number of conditions: {len(self.conditions)}")
        if len(self.product_classes) > 0:
            print(f"Number of product classes: {len(self.product_classes)}")

    def create_output_directory(self, subdir=None):
        """创建输出目录"""
        if subdir:
            directory = os.path.join(self.output_dir, subdir)
        else:
            directory = self.output_dir
            
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        return directory
    
    def visualize_rating_distribution(self):
        """可视化评分分布"""
        directory = self.create_output_directory("rating_stats")
        
        # 计算评分统计信息
        rating_mean = self.data['rating'].mean()
        rating_median = self.data['rating'].median()
        rating_std = self.data['rating'].std()
        rating_min = self.data['rating'].min()
        rating_max = self.data['rating'].max()
        
        # 创建评分直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='rating', bins=9, kde=True, discrete=True)
        
        plt.title("Rating Distribution")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        
        # 添加统计信息
        plt.figtext(0.15, 0.85, f"Mean: {rating_mean:.2f}\nMedian: {rating_median:.2f}\nStd Dev: {rating_std:.2f}\nMin: {rating_min:.2f}\nMax: {rating_max:.2f}")
        
        # 保存图表
        output_path = os.path.join(directory, "rating_histogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Rating distribution visualization saved to {output_path}")
        
        # 创建评分箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.data['rating'])
        
        plt.title("Rating Boxplot")
        plt.xlabel("Rating")
        plt.grid(axis='x', alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "rating_boxplot.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Rating boxplot saved to {output_path}")
    
    def visualize_usefulcount_distribution(self):
        """可视化有用评论数分布"""
        directory = self.create_output_directory("usefulcount_stats")
        
        # 计算有用评论数统计信息
        useful_mean = self.data['usefulCount'].mean()
        useful_median = self.data['usefulCount'].median()
        useful_std = self.data['usefulCount'].std()
        useful_min = self.data['usefulCount'].min()
        useful_max = self.data['usefulCount'].max()
        
        # 创建有用评论数直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='usefulCount', bins=50, discrete=True)
        
        plt.title("Useful Count Distribution")
        plt.xlabel("Useful Count")
        plt.ylabel("Frequency (Log Scale)")
        plt.grid(alpha=0.3)
        
        # 添加统计信息
        plt.figtext(0.7, 0.7, f"Mean: {useful_mean:.2f}\nMedian: {useful_median:.2f}\nStd Dev: {useful_std:.2f}\nMin: {useful_min:.2f}\nMax: {useful_max:.2f}")
        
        # 保存图表
        output_path = os.path.join(directory, "usefulcount_histogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Useful count distribution visualization saved to {output_path}")
        
        # 评分与有用评论数关系散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='rating', y='usefulCount', data=self.data, alpha=0.5)
        
        plt.title("Relationship Between Rating and Useful Count")
        plt.xlabel("Rating")
        plt.ylabel("Useful Count")
        plt.grid(alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "rating_vs_usefulcount.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Rating vs useful count relationship visualization saved to {output_path}")
    
    def visualize_time_distribution(self):
        """可视化时间分布"""
        directory = self.create_output_directory("time_stats")
        
        # 按年份分组
        yearly_counts = self.data.groupby(self.data['date'].dt.year).size().reset_index(name='count')
        
        # 创建年度评论数量折线图
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_counts['date'], yearly_counts['count'], marker='o', linestyle='-')
        
        plt.title("Yearly Review Count Trend")
        plt.xlabel("Year")
        plt.ylabel("Number of Reviews")
        plt.grid(True, alpha=0.3)
        plt.xticks(yearly_counts['date'])
        
        # 保存图表
        output_path = os.path.join(directory, "yearly_review_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Yearly review trend visualization saved to {output_path}")
        
        # 按月份分组
        monthly_counts = self.data.groupby(self.data['date'].dt.month).size().reset_index(name='count')
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly_counts['month_name'] = monthly_counts['date'].apply(lambda x: month_names[x-1])
        
        # 创建月度评论数量柱状图
        plt.figure(figsize=(12, 6))
        plt.bar(monthly_counts['month_name'], monthly_counts['count'])
        
        plt.title("Monthly Review Count Distribution")
        plt.xlabel("Month")
        plt.ylabel("Number of Reviews")
        plt.grid(axis='y', alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "monthly_review_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Monthly review distribution visualization saved to {output_path}")
        
        # 评分随时间变化趋势
        # 按月度分组计算平均评分
        monthly_ratings = self.data.groupby(pd.Grouper(key='date', freq='M'))['rating'].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        plt.plot(monthly_ratings['date'], monthly_ratings['rating'], linestyle='-')
        
        plt.title("Monthly Average Rating Trend")
        plt.xlabel("Date")
        plt.ylabel("Average Rating")
        plt.grid(True, alpha=0.3)
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # 保存图表
        output_path = os.path.join(directory, "monthly_rating_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Monthly rating trend visualization saved to {output_path}")
    
    def visualize_top_drugs(self, top_n=10):
        """可视化热门药品"""
        directory = self.create_output_directory("drug_stats")
        
        # 统计药品评论数量
        drug_counts = self.data['drugName'].value_counts().reset_index()
        drug_counts.columns = ['drugName', 'count']
        top_drugs = drug_counts.head(top_n)
        
        # 创建热门药品评论数量柱状图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_drugs['drugName'], top_drugs['count'], color='skyblue')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Drugs by Review Count")
        plt.xlabel("Review Count")
        plt.ylabel("Drug Name")
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()  # 让评论最多的药品显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_drugs_by_count.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top drugs by review count visualization saved to {output_path}")
        
        # 统计药品平均评分
        drug_ratings = self.data.groupby('drugName')['rating'].mean().reset_index()
        drug_ratings = drug_ratings.sort_values('rating', ascending=False)
        
        # 筛选评论数量大于10的药品
        drug_counts_dict = dict(zip(drug_counts['drugName'], drug_counts['count']))
        drug_ratings['count'] = drug_ratings['drugName'].map(drug_counts_dict)
        popular_drugs = drug_ratings[drug_ratings['count'] >= 10].head(top_n)
        
        # 创建高评分药品柱状图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(popular_drugs['drugName'], popular_drugs['rating'], color='lightgreen')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.2f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Drugs by Rating (Review Count ≥ 10)")
        plt.xlabel("Average Rating")
        plt.ylabel("Drug Name")
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()  # 让评分最高的药品显示在顶部
        plt.xlim(0, 10)  # 评分范围0-10
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_drugs_by_rating.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top rated drugs visualization saved to {output_path}")
        
        # 统计药品有用评论数
        drug_useful = self.data.groupby('drugName')['usefulCount'].sum().reset_index()
        drug_useful = drug_useful.sort_values('usefulCount', ascending=False)
        
        # 筛选评论数量大于10的药品
        drug_useful['count'] = drug_useful['drugName'].map(drug_counts_dict)
        useful_drugs = drug_useful.head(top_n)
        
        # 创建有用评论数最高药品柱状图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(useful_drugs['drugName'], useful_drugs['usefulCount'], color='coral')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Drugs by Total Useful Count")
        plt.xlabel("Total Useful Count")
        plt.ylabel("Drug Name")
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()  # 让有用评论数最高的药品显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_drugs_by_useful.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Most useful drugs visualization saved to {output_path}")
    
    def visualize_top_conditions(self, top_n=10):
        """可视化热门病症"""
        directory = self.create_output_directory("condition_stats")
        
        # 统计病症评论数量
        condition_counts = self.data['condition'].value_counts().reset_index()
        condition_counts.columns = ['condition', 'count']
        top_conditions = condition_counts.head(top_n)
        
        # 创建热门病症评论数量柱状图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_conditions['condition'], top_conditions['count'], color='skyblue')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:,.0f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Conditions by Review Count")
        plt.xlabel("Review Count")
        plt.ylabel("Condition")
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()  # 让评论最多的病症显示在顶部
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_conditions_by_count.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top conditions by review count visualization saved to {output_path}")
        
        # 统计病症平均评分
        condition_ratings = self.data.groupby('condition')['rating'].mean().reset_index()
        condition_ratings = condition_ratings.sort_values('rating', ascending=False)
        
        # 筛选评论数量大于10的病症
        condition_counts_dict = dict(zip(condition_counts['condition'], condition_counts['count']))
        condition_ratings['count'] = condition_ratings['condition'].map(condition_counts_dict)
        popular_conditions = condition_ratings[condition_ratings['count'] >= 10].head(top_n)
        
        # 创建高评分病症柱状图
        plt.figure(figsize=(12, 8))
        bars = plt.barh(popular_conditions['condition'], popular_conditions['rating'], color='lightgreen')
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            plt.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{width:.2f}', va='center', fontsize=9)
        
        plt.title(f"Top {top_n} Conditions by Rating (Review Count ≥ 10)")
        plt.xlabel("Average Rating")
        plt.ylabel("Condition")
        plt.grid(axis='x', alpha=0.3)
        plt.gca().invert_yaxis()  # 让评分最高的病症显示在顶部
        plt.xlim(0, 10)  # 评分范围0-10
        
        # 保存图表
        output_path = os.path.join(directory, f"top_{top_n}_conditions_by_rating.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Top rated conditions visualization saved to {output_path}")
    
    def visualize_product_class_distribution(self, top_n=10):
        """可视化产品类别分布"""
        if 'Product Class' not in self.data.columns or len(self.product_classes) == 0:
            print("Data does not contain product class information or all product classes are 'Else', skipping this visualization")
            return
            
        directory = self.create_output_directory("product_class_stats")
        
        # 过滤掉'Else'类别
        filtered_data = self.data[self.data['Product Class'] != 'Else']
        if filtered_data.empty:
            print("No product class data after filtering 'Else', skipping this visualization")
            return
            
        # 统计产品类别评论数量
        class_counts = filtered_data['Product Class'].value_counts().reset_index()
        class_counts.columns = ['Product Class', 'count']
        
        # 创建产品类别评论数量饼图
        plt.figure(figsize=(10, 8))
        
        # 计算百分比
        total_count = class_counts['count'].sum()
        percentages = [(count / total_count) * 100 for count in class_counts['count']]
        
        # 创建自定义标签函数
        def my_autopct(pct):
            index = percentages.index(min(percentages, key=lambda x: abs(x-pct)))
            val = class_counts['count'].iloc[index]
            return f"{pct:.1f}%\n({val:,.0f})"
        
        plt.pie(class_counts['count'], labels=class_counts['Product Class'], 
                autopct=my_autopct, startangle=90, shadow=True, 
                explode=[0.05] * len(class_counts),
                textprops={'fontsize': 12})
        
        plt.axis('equal')  # 确保饼图是圆的
        plt.title("Product Class Distribution")
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_distribution_pie.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class distribution visualization saved to {output_path}")
        
        # 统计产品类别平均评分
        class_ratings = filtered_data.groupby('Product Class')['rating'].mean().reset_index()
        class_ratings = class_ratings.sort_values('rating', ascending=False)
        
        # 创建产品类别评分柱状图
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_ratings['Product Class'], class_ratings['rating'], color='lightgreen')
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height,
                     f'{height:.2f}', ha='center', va='bottom', rotation=0, fontsize=9)
        
        plt.title("Average Rating by Product Class")
        plt.xlabel("Product Class")
        plt.ylabel("Average Rating")
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(0, 10)  # 评分范围0-10
        plt.xticks(rotation=45)
        
        # 保存图表
        output_path = os.path.join(directory, "product_class_rating.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Product class rating visualization saved to {output_path}")
    
    def visualize_review_length_stats(self):
        """可视化评论长度统计"""
        directory = self.create_output_directory("review_stats")
        
        # 计算评论长度统计信息
        length_mean = self.data['review_length'].mean()
        length_median = self.data['review_length'].median()
        length_std = self.data['review_length'].std()
        length_min = self.data['review_length'].min()
        length_max = self.data['review_length'].max()
        
        # 创建评论长度直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='review_length', bins=200)
        
        plt.title("Review Length Distribution")
        plt.xlabel("Review Length (characters)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        
        # 添加统计信息
        plt.figtext(0.7, 0.7, f"Mean: {length_mean:.2f}\nMedian: {length_median:.2f}\nStd Dev: {length_std:.2f}\nMin: {length_min:.2f}\nMax: {length_max:.2f}")
        
        # 保存图表
        output_path = os.path.join(directory, "review_length_histogram.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Review length distribution visualization saved to {output_path}")
        
        # 评论长度与评分关系散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='rating', y='review_length', data=self.data, alpha=0.3)
        
        plt.title("Relationship Between Rating and Review Length")
        plt.xlabel("Rating")
        plt.ylabel("Review Length (characters)")
        plt.grid(alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "rating_vs_length.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Rating vs review length relationship visualization saved to {output_path}")
        
        # 评论长度与有用评论数关系散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='review_length', y='usefulCount', data=self.data, alpha=0.3)
        
        plt.title("Relationship Between Review Length and Useful Count")
        plt.xlabel("Review Length (characters)")
        plt.ylabel("Useful Count")
        plt.grid(alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(directory, "length_vs_usefulcount.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Review length vs useful count relationship visualization saved to {output_path}")
    
    def generate_wordcloud(self, top_n=None):
        """生成评论词云"""
        directory = self.create_output_directory("wordcloud")
        
        if top_n:
            # 获取评论数最多的top_n个药品
            drug_counts = self.data['drugName'].value_counts().head(top_n)
            top_drugs = drug_counts.index.tolist()
            
            for drug in top_drugs:
                drug_reviews = self.data[self.data['drugName'] == drug]['review']
                
                # 合并所有评论
                text = " ".join([str(review) for review in drug_reviews if pd.notna(review)])
                
                # 清洗文本
                text = re.sub(r'[^\w\s]', '', text)
                text = re.sub(r'\d+', '', text)
                
                # 生成词云
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=200,
                    contour_width=3,
                    contour_color='steelblue'
                ).generate(text)
                
                # 显示词云
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title(f"{drug} Review Word Cloud")
                
                # 保存图表
                output_path = os.path.join(directory, f"wordcloud_{drug}.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Word cloud for {drug} saved to {output_path}")
        
        # 所有评论词云
        # 取样以减轻计算负担
        sample_size = min(10000, len(self.data))
        sample_reviews = self.data['review'].sample(sample_size)
        
        # 合并评论
        text = " ".join([str(review) for review in sample_reviews if pd.notna(review)])
        
        # 清洗文本
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # 生成词云
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=200,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        # 显示词云
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("All Reviews Word Cloud")
        
        # 保存图表
        output_path = os.path.join(directory, "wordcloud_all.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Word cloud for all reviews saved to {output_path}")
    
    def visualize_drug_comparison(self, drug_list=None, top_n=5):
        """比较不同药品的评分和有用评论数"""
        directory = self.create_output_directory("drug_comparison")
        
        if drug_list is None:
            # 获取评论数最多的top_n个药品
            drug_counts = self.data['drugName'].value_counts().head(top_n)
            drug_list = drug_counts.index.tolist()
        
        # 筛选数据
        filtered_data = self.data[self.data['drugName'].isin(drug_list)]
        
        # 比较药品评分
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='drugName', y='rating', data=filtered_data)
        
        plt.title(f"Top {top_n} Drug Rating Comparison")
        plt.xlabel("Drug Name")
        plt.ylabel("Rating")
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图表
        output_path = os.path.join(directory, "drug_rating_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Drug rating comparison visualization saved to {output_path}")
        
        # 比较药品评分随时间变化
        plt.figure(figsize=(12, 6))
        
        for drug in drug_list:
            drug_data = filtered_data[filtered_data['drugName'] == drug]
            drug_ratings = drug_data.groupby(pd.Grouper(key='date', freq='QE'))['rating'].mean().reset_index()
            plt.plot(drug_ratings['date'], drug_ratings['rating'], linestyle='-', label=drug)
        
        plt.title(f"Top {top_n} Drug Rating Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel("Average Rating")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # 保存图表
        output_path = os.path.join(directory, "drug_rating_time_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Drug rating time trend visualization saved to {output_path}")

    def visualize_condition_comparison(self, condition_list=None, top_n=5):
        """比较不同病症的评分和有用评论数"""
        directory = self.create_output_directory("condition_comparison")
        
        if condition_list is None:
            # 获取评论数最多的top_n个病症
            condition_counts = self.data['condition'].value_counts().head(top_n)
            condition_list = condition_counts.index.tolist()
        
        # 筛选数据
        filtered_data = self.data[self.data['condition'].isin(condition_list)]
        
        # 比较病症评分
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='condition', y='rating', data=filtered_data)
        
        plt.title(f"Top {top_n} Condition Rating Comparison")
        plt.xlabel("Condition Name")
        plt.ylabel("Rating")
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        
        # 保存图表
        output_path = os.path.join(directory, "condition_rating_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Condition rating comparison visualization saved to {output_path}")
        
        # 为每个病症找出最常用药品
        condition_top_drugs = {}
        for condition in condition_list:
            condition_data = filtered_data[filtered_data['condition'] == condition]
            top_drugs = condition_data['drugName'].value_counts().head(3)
            condition_top_drugs[condition] = top_drugs
        
        # 创建病症常用药品图表
        fig, axes = plt.subplots(len(condition_list), 1, figsize=(10, 5 * len(condition_list)))
        
        if len(condition_list) == 1:
            axes = [axes]
            
        for i, condition in enumerate(condition_list):
            top_drugs = condition_top_drugs[condition]
            ax = axes[i]
            bars = ax.barh(top_drugs.index, top_drugs.values, color=f'C{i}')
            
            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                ax.text(width + width * 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:,.0f}', va='center', fontsize=9)
            
            ax.set_title(f"{condition} - Common Drugs")
            ax.set_xlabel("Review Count")
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(directory, "condition_top_drugs.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Condition common drugs visualization saved to {output_path}")
    
    def run_all_visualizations(self):
        """执行所有可视化"""
        print("Starting all visualizations...")
        self.visualize_rating_distribution()
        self.visualize_usefulcount_distribution()
        self.visualize_time_distribution()
        self.visualize_top_drugs()
        self.visualize_top_conditions()
        self.visualize_product_class_distribution()
        self.visualize_review_length_stats()
        self.generate_wordcloud()
        self.visualize_drug_comparison()
        self.visualize_condition_comparison()
        
        print("All visualizations completed!")