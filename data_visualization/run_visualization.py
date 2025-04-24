from pharm_data_visualization import PharmDataVisualization
from review_data_visualization import ReviewDataVisualization


# 初始化可视化类
pharm_visualizer = PharmDataVisualization(
    data_path="./data/Pharm Data_Data.csv",
    output_dir="./data_visualization/pharm_visualization_results",
)

pharm_visualizer.run_all_visualizations()
pharm_visualizer.run_all_single_visualizations()

review_visualizer = ReviewDataVisualization(
    data_path="./data/drugsComTrain_raw_addclass.csv",
    output_dir="./data_visualization/review_visualization_results",
)

review_visualizer.run_all_visualizations()

