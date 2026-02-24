📌 1. Project Overview ｜ 项目概述

🎯 Objective: Predict whether a ride booking will be cancelled using two different models: Logistic Regression(@Shen Ziyun) and XGBoost(@Janice Lam). 
Build two models by Logistic Regression and XGBoost seperately to see which model is more suitable to this project

项目目标：使用 Logistic Regression(@Shen Ziyun) 和 XGBoost(@Janice Lam) 分别构建模型，预测打车订单是否会被取消，并研究哪个模型更适合该项目

🚗 Context: Ride-hailing platforms face high uncertainty due to customer cancellations
业务背景：打车平台面临较高的订单取消不确定性

📊 Focus: Emphasize feature engineering, interpretability, and realistic deployment settings
核心关注点：特征工程、模型可解释性以及真实业务场景下的预测能力

📂 2. Dataset ｜ 数据集说明

📄 Source: NCR ride booking dataset from Kaggle: https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard
数据来源：Kaggle 上的NCR地区打车订单原始数据

🗂 Raw data only (no pre-engineered CSVs used)
全程使用 原始 CSV 文件，未依赖预处理好的 engineered 数据

🧾 Key fields include:
Booking Status/Date & Time/Pickup locations/Drop locations/Waiting time metrics (e.g. Avg VTAT, Avg CTAT)
主要字段包括：订单状态, 日期与时间, 上车/下车地点, 等待时间相关指标（如：平均等待时长等）

🧹 3. Data Preparation & Feature Engineering ｜ 数据清洗与特征工程

🧼 Filtered to five relevant booking statuses:
Cancelled by Customer/Completed/Cancelled by Driver/Incomplete/No Driver Found
保留五种核心订单状态：用户取消，已完成，司机取消，未完成与未找到司机

🎯 Target variable 目标变量定义:
Is_Cancelled = 1 if booking is not completed，表示订单未完成
Is_Cancelled = 0 otherwise，表示订单成功完成

⏰ Time features extracted: Imported a calendar package to support time-based feature extraction
Hour of day/Weekday/Weekend/Peak-hour categorize
时间特征构造：引入日历相关 package 以支持时间维度特征构造
包括：小时，工作日/周末，上下班高峰时段分类

📍 Location features:
Pickup/Drop frequency/Encoded pickup & drop locations/Same-area indicator
地点特征：上车 / 下车地点出现频次，地点编码，是否为同一区域

🧩 Missing values:
Avg VTAT and Avg CTAT imputed using mean values
缺失值处理：
对 Avg VTAT 与 Avg CTAT 使用均值填补

🤖 4. Modeling ｜ 模型构建

🌲 Model: XGBoost Classifier(Me) 使用模型：XGBoost 分类器

🔧 Design choices:
Grouping the training set in the same proportions as in the Logistic Regression
Shallow trees with regularization to prevent overfitting
No SMOTE applied (class imbalance handled implicitly)
设计原则：
与Logistics Regression保持相同比例的training set分组
使用较浅的树结构并加入正则化，避免过拟合
未使用 SMOTE，避免引入人工合成样本

📦 Feature set:
Pre-booking features, Avg CTAT as a strong predictive signal
特征组合：
下单前可获取的基础特征, Avg CTAT 作为关键信号特征

📈 5. Evaluation ｜ 模型评估

📊 Train / Test split: 70% / 30% (stratified)

🧪 Evaluation metrics:
ROC-AUC
PR-AUC
Confusion Matrix

⭐ Final performance:
ROC-AUC ≈ 0.97
PR-AUC ≈ 0.97
High precision with strong recall on cancelled orders

📊 数据划分：训练集 / 测试集 = 7 : 3（分层抽样）

🧪 评估指标：
ROC-AUC
PR-AUC
混淆矩阵

⭐ 最终效果：
ROC-AUC ≈ 0.97
PR-AUC ≈ 0.97
对取消订单具有较高的识别能力

💡 6. Key Insights ｜ 核心结论

🔑 Actual waiting time (Avg CTAT) is the dominant driver of cancellations 实际等待时间（Avg CTAT）是影响订单取消的最关键因素

📉 Models using only pre-booking information show limited predictability 仅使用下单前信息时，取消行为本身较难预测

🧠 Model performance is primarily constrained by information availability, not algorithm choice 模型效果的上限主要由信息质量决定，而非算法复杂度

💡Practical Implications for Business ｜ 业务建议

🔹 1. Proactive Risk Intervention: Use real-time prediction scores to identify high-risk orders and intervene before cancellation occurs 风险预防：使用实时预测评分识别高风险订单，并在取消发生前进行干预

🔹 2. Operational Efficiency Optimization: Minimize resource waste caused by cancellations 效率优化： 最大限度减少因取消造成的资源浪费，eg：减少无效派单与空驶里程

🔹 3. Data-Driven Strategic Improvement: Leverage feature importance insights for long-term platform optimization 长期优化：利用特征重要性洞察实现平台长期优化，eg：基于特征重要性分析优化调度策略










