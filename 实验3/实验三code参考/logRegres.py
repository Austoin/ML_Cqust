import numpy as np
import matplotlib.pyplot as plt
import random

class HorseColicPredictor:
    # 一个封装了逻辑回归实验全过程的类。
    # 遵循PDF文档要求，包括均值填充预处理，并简化了终端输出。
    def __init__(self):
        # 初始化，用于存储训练集的特征均值
        self.training_means = None

    @staticmethod
    def sigmoid(inX):
        # Sigmoid函数
        inX = np.clip(inX, -500, 500)
        return 1.0 / (1 + np.exp(-inX))

    def train_bga(self, data_matrix, class_labels, alpha=0.001, max_cycles=500):
        # 批量梯度上升 (BGA)，用于可视化
        data_matrix = np.array(data_matrix)
        label_mat = np.array(class_labels).reshape(-1, 1)
        weights = np.ones((data_matrix.shape[1], 1))
        for _ in range(max_cycles):
            h = self.sigmoid(data_matrix @ weights)
            error = label_mat - h
            weights = weights + alpha * data_matrix.T @ error
        return weights

    def train_sgd(self, data_matrix, class_labels, num_iter=150):
        # 随机梯度上升 (SGD)，用于主模型训练
        data_matrix = np.array(data_matrix)
        m, n = data_matrix.shape
        weights = np.ones(n)
        for j in range(num_iter):
            data_index = list(range(m))
            for i in range(m):
                # 动态学习率
                alpha = 4 / (1.0 + j + i) + 0.0001
                rand_idx = random.choice(data_index)
                h = self.sigmoid(np.sum(data_matrix[rand_idx] * weights))
                error = class_labels[rand_idx] - h
                weights = weights + alpha * error * data_matrix[rand_idx]
                data_index.remove(rand_idx)
        return weights

    def classify(self, inX, weights):
        # 分类函数
        prob = self.sigmoid(np.sum(inX * weights))
        return 1.0 if prob > 0.5 else 0.0

    def load_data(self, filename, is_simple_data=False):
        # 通用数据加载函数
        feature_set, label_set = [], []
        with open(filename) as f:
            for line in f.readlines():
                parts = line.strip().split()
                if is_simple_data:
                    feature_set.append([1.0, float(parts[0]), float(parts[1])])
                    label_set.append(int(parts[2]))
                else:
                    feature_set.append([float(val) for val in parts[:-1]])
                    label_set.append(float(parts[-1]))
        return feature_set, label_set

    def preprocess_mean_fill(self, dataset, is_training=True):
        # 预处理：使用训练集的均值填充缺失值(0)
        feature_matrix = np.array(dataset)
        if is_training:
            num_features = feature_matrix.shape[1]
            means = np.zeros(num_features)
            for i in range(num_features):
                non_zero_vals = feature_matrix[:, i][feature_matrix[:, i] != 0]
                if len(non_zero_vals) > 0:
                    means[i] = np.mean(non_zero_vals)
            self.training_means = means
        
        for i in range(feature_matrix.shape[1]):
            zero_indices = np.where(feature_matrix[:, i] == 0)[0]
            feature_matrix[zero_indices, i] = self.training_means[i]
        return feature_matrix

    def run_visualization(self):
        # 分析数据 - 可视化
        print("算法原理可视化...")
        data_arr, label_mat = self.load_data('testSet.txt', is_simple_data=True)
        weights = self.train_bga(data_arr, label_mat)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(np.array(data_arr)[np.array(label_mat)==1, 1], np.array(data_arr)[np.array(label_mat)==1, 2], s=30, c='red', marker='s', label='类别 1')
        ax.scatter(np.array(data_arr)[np.array(label_mat)==0, 1], np.array(data_arr)[np.array(label_mat)==0, 2], s=30, c='green', label='类别 0')
        
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0] - weights[1] * x) / weights[2]
        ax.plot(x, y.flatten())
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title('逻辑回归决策边界'); plt.xlabel('特征 X1'); plt.ylabel('特征 X2')
        plt.legend(); plt.show()
        print("可视化完成。\n")

    def test_model(self, num_iter=1000):
        # 训练并测试模型
        training_set, training_labels = self.load_data('horseColicTraining.txt')
        test_set, test_labels = self.load_data('horseColicTest.txt')
        
        processed_training_set = self.preprocess_mean_fill(training_set, is_training=True)
        processed_test_set = self.preprocess_mean_fill(test_set, is_training=False)
        
        weights = self.train_sgd(processed_training_set, training_labels, num_iter)

        error_count = sum(1 for i, vec in enumerate(processed_test_set) if int(self.classify(vec, weights)) != int(test_labels[i]))
        
        return float(error_count) / len(test_labels), weights

    def run_multi_tests(self, num_tests=10, num_iter=1000):
        # 多次测试，打印每次的错误率，并计算平均值
        print("马疝病预测 (均值填充法)...")
        error_rates = []
        for i in range(num_tests):
            error_rate, _ = self.test_model(num_iter)
            print(f"第 {i+1} 次测试错误率: {error_rate:.4f}")
            error_rates.append(error_rate)
        
        avg_error = np.mean(error_rates)
        print(f"\n{num_tests}次测试平均错误率: {avg_error:.4f}\n")

    def compare_with_linear_reg(self):
        # 与线性回归对比
        print("模型对比 (vs. 线性回归)...")
        training_set, training_labels = self.load_data('horseColicTraining.txt')
        test_set, test_labels = self.load_data('horseColicTest.txt')
        
        X_train = self.preprocess_mean_fill(training_set, is_training=True)
        X_test = self.preprocess_mean_fill(test_set, is_training=False)
        y_train = np.array(training_labels)

        X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
        try:
            weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
        except np.linalg.LinAlgError:
            print("线性回归失败: 矩阵为奇异矩阵。")
            return

        X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        predictions = X_test_b @ weights
        
        error_count = sum(1 for i, p in enumerate(predictions) if (1 if p > 0.5 else 0) != test_labels[i])
        error_rate = error_count / len(test_labels)
        print(f"线性回归错误率: {error_rate:.4f}\n")

    def run_cli_predictor(self, weights):
        # 使用算法 - 命令行预测工具
        print("命令行预测...")
        print("输入21项指标:")
        
        while True:
            line = input("> ")
            if line.lower() == 'quit':
                print("已退出。"); break
            
            try:
                features = np.array([float(p) for p in line.strip().split()])
                if len(features) != 21:
                    print(f"错误: 需要输入21个指标, 您输入了{len(features)}个。")
                    continue
                
                processed_features = self.preprocess_mean_fill([features], is_training=False)
                prediction = self.classify(processed_features[0], weights)
                print(f"预测结果: {'死亡' if prediction == 1.0 else '存活'}")

            except ValueError: print("错误: 请确保所有输入都是有效的数字。")

if __name__ == '__main__':
    predictor = HorseColicPredictor()
    predictor.run_visualization()
    predictor.run_multi_tests(num_iter=1000)
    predictor.compare_with_linear_reg()
    _, final_weights = predictor.test_model(num_iter=1000)
    predictor.run_cli_predictor(final_weights)