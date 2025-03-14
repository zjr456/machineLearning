# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:36:07 2024

@author: BIENVENUE
"""
from sklearn import tree
import os
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import tree

# 将预测结果保存为向量
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  confusion_matrix, classification_report
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support







def load_data(folder_path):
    all_data = []
    # 遍历文件夹中的所有 .mat 文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mat'):
            file_path = os.path.join(folder_path, file_name)
            mat_data = loadmat(file_path)  # 加载 .mat 文件
            # 提取信号数据（假设数据存储在 'data' 键中）
            signals = mat_data['d_iner']  
            # 提取动作编号、参与者编号和试验编号
            action, subject, trial = parse_filename(file_name)
            # 将数据存入数据框
            df = pd.DataFrame(signals, columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz'])
            df['subject'] = subject
            df['trial'] = trial
            df['action'] = action
            all_data.append(df)
    # 合并所有数据
    return pd.concat(all_data, ignore_index=True)

def parse_filename(file_name):
    parts = file_name.split('_')
    action = int(parts[0][1:])
    subject = int(parts[1][1:])
    trial = int(parts[2][1:])
    return action, subject, trial




def plot_signal(dataframe, sensor, action, subject, trial):
    # 过滤出特定动作、参与者和试验的数据
    filtered_data = dataframe[
        (dataframe['action'] == action) & 
        (dataframe['subject'] == subject) & 
        (dataframe['trial'] == trial)
    ]
    # 确定信号类型（加速度计或陀螺仪）
    if sensor == 1:  # 加速度计
        columns = ['Ax', 'Ay', 'Az']
        title = 'Accelerometer Signal'
    elif sensor == 2:  # 陀螺仪
        columns = ['Gx', 'Gy', 'Gz']
        title = 'Gyroscope Signal'
    
    # 绘图
    plt.figure(figsize=(10, 6))
    for col in columns:
        plt.plot(filtered_data[col], label=col)
    plt.title(f'{title} for Action {action}, Subject {subject}, Trial {trial}')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()
    

def feature_extraction(dataframe):
    # 按动作编号分组
    grouped = dataframe.groupby(['action', 'subject', 'trial'])
    features = []

    for (action_id, subject_id, trial_id), group in grouped:
        # 提取信号列
        signals = group[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']]
        
        # 计算特征
        feature_vector = [
            signals['Ax'].mean(), signals['Ax'].std(),
            signals['Ay'].mean(), signals['Ay'].std(),
            signals['Az'].mean(), signals['Az'].std(),
            signals['Gx'].mean(), signals['Gx'].std(),
            signals['Gy'].mean(), signals['Gy'].std(),
            signals['Gz'].mean(), signals['Gz'].std(),
        ]
        
        # 添加动作编号、参与者编号、试验编号
        feature_vector.extend([action_id, subject_id, trial_id])
        
        # 将特征向量添加到特征列表
        features.append(feature_vector)

    # 创建新的 DataFrame
    columns = [
        'mean_Ax', 'std_Ax', 'mean_Ay', 'std_Ay', 'mean_Az', 'std_Az',
        'mean_Gx', 'std_Gx', 'mean_Gy', 'std_Gy', 'mean_Gz', 'std_Gz',
        'action', 'subject', 'trial'
    ]
    return pd.DataFrame(features, columns=columns)


def visualize_bar(features_df, feature_column):
    plt.figure(figsize=(10, 6))
    plt.bar(features_df['action'], features_df[feature_column])
    plt.xlabel('Action')
    plt.ylabel(feature_column)
    plt.title(f'{feature_column} for each Action')
    plt.show()

def visualize_scatter(features_df, feature_x, feature_y):
    plt.figure(figsize=(10, 6))
    for action_id in features_df['action'].unique():
        subset = features_df[features_df['action'] == action_id]
        plt.scatter(subset[feature_x], subset[feature_y], label=f'Action {action_id}')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'{feature_x} vs {feature_y}')
    plt.legend()
    plt.show()


def prepare_data(features_df):
    # 划分训练集和测试集
    train_subjects = [1, 3, 5, 7]
    test_subjects = [2, 4, 6, 8]
    
    # 训练集
    train_data = features_df[features_df['subject'].isin(train_subjects)]
    train_labels = train_data['action']
    train_features = train_data.drop(columns=['action', 'subject', 'trial'])  # 只保留特征列
    
    # 测试集
    test_data = features_df[features_df['subject'].isin(test_subjects)]
    test_labels = test_data['action']
    test_features = test_data.drop(columns=['action', 'subject', 'trial'])  # 只保留特征列

    # 归一化：基于训练集计算均值和标准差
    mean_vector = train_features.mean()
    std_vector = train_features.std()

    train_features_normalized = (train_features - mean_vector) / std_vector
    test_features_normalized = (test_features - mean_vector) / std_vector

    # 返回结果
    return train_features_normalized, train_labels, test_features_normalized, test_labels

 
    
def Classification_model_tree(train_data,train_labels, test_data, test_labels):
    # 初始化决策树分类器
    clf = DecisionTreeClassifier()
    
    # 训练模型
    clf.fit(train_data, train_labels)
    # 绘制决策树
    plt.figure(figsize=(12, 8))
    plot_tree(clf, filled=True, feature_names=train_data.columns, class_names=[str(i) for i in range(1, 28)])
    plt.title("Decision Tree Visualization")
    plt.show()
    # 导出决策树规则
    tree_rules = export_text(clf, feature_names=list(train_data.columns))
    #print(tree_rules)
    
    # 预测测试集
    test_predictions = clf.predict(test_data)
    # 评估模型性能
    accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {accuracy:.2f}")
    importance_of_tree(train_data, train_labels)



def importance_of_tree(train_data, train_labels):
    # 决策树分类器训练
    clf = DecisionTreeClassifier()
    clf.fit(train_data, train_labels)

    # 获取特征重要性
    feature_importances = pd.DataFrame({
        'Feature': train_data.columns,
        'Importance': clf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # 打印特征重要性
    print("Feature Importance:\n", feature_importances)
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance in Decision Tree')
    plt.gca().invert_yaxis()
    plt.show()
    
    mode_test(test_data,test_labels,clf)
    
    return clf



def Classification_model_svc(train_data, train_labels, test_data, test_labels):
    # 初始化分类器
    svm_clf = SVC(kernel='linear')

    # 训练模型
    svm_clf.fit(train_data, train_labels)

    # 评估模型
    svm_accuracy = svm_clf.score(test_data, test_labels)
    print(f"SVM Test Accuracy: {svm_accuracy:.2f}")
    mode_test(test_data,test_labels,svm_clf)
    return svm_clf


def Classification_model_KNN(train_data, train_labels, test_data, test_labels):
    # 初始化分类器
    knn_clf = KNeighborsClassifier(n_neighbors=5)

    # 训练模型
    knn_clf.fit(train_data, train_labels)

    # 评估模型
    knn_accuracy = knn_clf.score(test_data, test_labels)
    print(f"KNN Test Accuracy: {knn_accuracy:.2f}")
    mode_test(test_data,test_labels,knn_clf)
    return knn_clf


def Classification_model_Selector(train_data, train_labels, test_data, test_labels,id):
    if id ==0:
        clf=Classification_model_tree(train_data, train_labels, test_data, test_labels)
        
    elif id ==1:
        clf=Classification_model_svc(train_data, train_labels, test_data, test_labels)
        
    elif id == 2:
        clf=Classification_model_KNN(train_data, train_labels, test_data, test_labels)
        
        
        

def mode_test(test_data,test_labels,clf):

    # 使用测试集进行预测
    predicted_labels = clf.predict(test_data)
    
    # 打印预测结果
    #print("Predicted Labels:\n", predicted_labels)
    
    # 计算模型准确率
    
    accuracy = accuracy_score(test_labels, predicted_labels)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    # 打印混淆矩阵和分类报告
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    #print("Confusion Matrix:\n", conf_matrix)
    
    class_report = classification_report(test_labels, predicted_labels)
    #print("Classification Report:\n", class_report)
    
    predicted_labels_vector = np.array(predicted_labels)
    print("Predicted Labels Vector:\n", predicted_labels_vector)
    
    # 调用函数
    analyze_model_performance(test_labels, predicted_labels)
    


def analyze_model_performance(test_labels, predicted_labels):
    # 分类报告
    print("Classification Report:\n")
    
    print(classification_report(test_labels, predicted_labels, target_names=[f"Class {i}" for i in range(1, 28)]))

    # 混淆矩阵
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    print("Confusion Matrix:\n", conf_matrix)
    
    
    
    report = classification_report(
    test_labels,
    predicted_labels,
    target_names=[f"Class {i}" for i in range(1, 28)],
    output_dict=True  # 转换为字典
    )
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.iloc[:-3, :]  # 移除最后三行

    
    
    #plot precision recall f1 score
    plot_precision_Recall_F1_Score(report_df)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"Class {i}" for i in range(1, 28)],
                yticklabels=[f"Class {i}" for i in range(1, 28)])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def plot_precision_Recall_F1_Score(report_df):
    # 提取数据
    classes = report_df.index
    precision = report_df['precision']
    recall = report_df['recall']
    f1_score = report_df['f1-score']
    
    # 设置图形
    x = np.arange(len(classes))  # 类别数量
    width = 0.25  # 柱状图宽度
    
    # 创建柱状图
    plt.figure(figsize=(15, 8))
    plt.bar(x - width, precision, width, label='Precision', color='skyblue')
    plt.bar(x, recall, width, label='Recall', color='lightgreen')
    plt.bar(x + width, f1_score, width, label='F1-Score', color='salmon')
    
    # 设置图例和标签
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Precision, Recall, and F1-Score for Each Class')
    plt.xticks(x, classes, rotation=90)
    plt.legend()
    plt.tight_layout()
    
    # 显示图形
    plt.show()


        

    
# 数据路径
folder_path = 'D:\machine_learning\IMU\IMU'
# 加载数据
data = load_data(folder_path)
#print(data.columns)
plot_signal(data, sensor=2, action=1, subject=1, trial=1)

columns = [
    'mean_Ax', 'std_Ax', 'mean_Ay', 'std_Ay', 'mean_Az', 'std_Az',
    'mean_Gx', 'std_Gx', 'mean_Gy', 'std_Gy', 'mean_Gz', 'std_Gz',
    'action', 'subject', 'trial'
]
# 假设 dataframe 是之前加载的原始数据
features_df = feature_extraction(data)

visualize_bar(features_df,'action')

# 准备数据
train_data, train_labels, test_data, test_labels = prepare_data(features_df)
Classification_model_Selector(train_data, train_labels, test_data, test_labels,0)
print('-----------------------------------------next method----------------------------------')
Classification_model_Selector(train_data, train_labels, test_data, test_labels,1)
print('-----------------------------------------next method----------------------------------')
Classification_model_Selector(train_data, train_labels, test_data, test_labels,2)



#
# 可视化动作 1，参与者 1，试验 1 的加速度计信号
#question 6
#




# 查看归一化后的数据
print("训练数据：", train_data.head())
print("训练标签：", train_labels.head())
print("测试数据：", test_data.head())
print("测试标签：", test_labels.head())











