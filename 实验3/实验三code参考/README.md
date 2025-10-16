Jupyter Notebook: 使用逻辑回归预测马疝病的死亡率
简介:
这个 Notebook 将引导你完成一个经典的机器学习项目：使用逻辑回归算法，根据马的临床数据来预测其是否会因疝气病症而死亡。

我们将分两大部分进行：

第一部分：算法原理与可视化 - 我们会先用一个非常简单的数据集来学习逻辑回归的基本原理，并画出它的分类边界，让你有一个直观的感受。

第二部分：实战马疝病数据预测 - 我们将使用真实的马疝病数据集，应用一个更高效的算法版本来训练模型，并评估其预测的准确性。

准备工作:
请确保你的环境中已经安装了 numpy 和 matplotlib 库。如果没有，请运行：
pip install numpy matplotlib

同时，请确保 testSet.txt, horseColicTraining.txt, 和 horseColicTest.txt 这三个数据文件与本 Notebook 文件在同一个文件夹下。

单元格 1: 导入必要的库
讲解:
这是我们的第一步，导入项目所需的 Python 库。

numpy: 这是 Python 中用于科学计算的核心库，我们将用它来处理矩阵和向量运算，这对于机器学习算法至关重要。我们通常将其简写为 np。

matplotlib.pyplot: 这是一个强大的绘图库，我们将用它来将我们的数据和模型结果可视化。我们通常将其简写为 plt。

code
Python
# 单元格 1: 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
第一部分: 算法原理与可视化
讲解:
在处理复杂的马疝病数据之前，我们先用一个简单的人造数据集 testSet.txt 来理解逻辑回归是如何工作的。这个数据集只有两个特征，方便我们在二维平面上展示。

单元格 2: 定义辅助函数 (加载数据与Sigmoid函数)
讲解:
这里我们定义两个基础函数：

loadDataSet(): 这个函数负责读取 testSet.txt 文件。文件中的每一行包含两个特征值和一个类别标签（0 或 1）。函数会把它们解析出来，并在特征前面加上一个恒为 1.0 的 X0 项，这是为了方便后续的矩阵运算。

sigmoid(inX): 这是逻辑回归算法的核心函数。它的作用是将任何输入的实数值“压缩”到 0 和 1 之间，输出的结果可以被看作是一个概率。例如，输出 0.8 意味着模型预测有 80% 的概率属于类别 1。

code
Python
# 单元格 2: 定义辅助函数
def loadDataSet():
    """加载并解析 testSet.txt 数据。"""
    dataMat = []
    labelMat = []
    with open('testSet.txt') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            # X0 设为 1.0，方便计算截距
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    """Sigmoid 函数"""
    # 使用 np.clip 防止指数运算溢出
    inX = np.clip(inX, -700, 700)
    return 1.0 / (1 + np.exp(-inX))
单元格 3: 定义梯度上升优化算法
讲解:
这个 gradAscent 函数实现了批量梯度上升算法。它的目标是为每个特征找到一个最佳的权重 (weight)，这些权重组合起来就构成了我们的分类模型。

它的工作流程如下：

将输入数据转换成 NumPy 矩阵，方便计算。

初始化一组权重，可以从全 1 开始。

在设定的迭代次数（maxCycles）内循环：
a. 使用当前的权重和 Sigmoid 函数计算每个数据点的预测概率。
b. 计算预测概率与真实标签之间的误差。
c. 根据这个误差，沿着梯度的方向微调所有的权重。学习率 alpha 控制了每次调整的幅度。

循环结束后，返回训练好的最佳权重。

code
Python
# 单元格 3: 定义梯度上升优化算法
def gradAscent(dataMatIn, classLabels):
    """批量梯度上升算法"""
    dataMatrix = np.array(dataMatIn)
    # 将标签列表转换为列向量
    labelMat = np.array(classLabels).reshape(-1, 1)
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # 学习率
    maxCycles = 500  # 迭代次数
    weights = np.ones((n, 1)) # 初始化权重

    for k in range(maxCycles):
        # 核心公式: 矩阵乘法计算预测值 h
        h = sigmoid(dataMatrix @ weights)
        # 计算误差
        error = (labelMat - h)
        # 更新权重
        weights = weights + alpha * dataMatrix.T @ error
        
    return weights
单元格 4: 定义可视化函数
讲解:
这个 plotBestFit 函数的作用是将我们的努力成果画出来。它会做两件事：

将 testSet.txt 中的数据点根据它们的类别（0 或 1）用不同颜色（绿色和红色）画在图上。

利用我们刚刚通过梯度上升算法计算出的最佳权重，画出一条决策边界。这条线就是我们模型的分类标准：线一侧的点被预测为类别 0，另一侧被预测为类别 1。

code
Python
# 单元格 4: 定义可视化函数
def plotBestFit(weights):
    """可视化数据集和逻辑回归的最佳拟合直线"""
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    
    # 将数据点按类别分组
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(len(labelMat)):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
            
    # 创建画布并绘制散点图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s', label='类别 1')
    ax.scatter(xcord2, ycord2, s=30, c='green', label='类别 0')
    
    # 绘制决策边界线
    x = np.arange(-3.0, 3.0, 0.1)
    # 方程: w0 + w1*x + w2*y = 0  =>  y = (-w0 - w1*x) / w2
    y = (-weights[0] - weights[1] * x) / weights[2]
    
    ax.plot(x, y.flatten())
    
    # 设置图表中文标题和标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('逻辑回归最佳拟合直线')
    plt.xlabel('特征 X1')
    plt.ylabel('特征 X2')
    plt.legend()
    plt.show()
单元格 5: 运行并查看可视化结果
讲解:
现在，我们将上面定义的函数串联起来。我们加载数据，用 gradAscent 训练模型得到权重，然后用 plotBestFit 将结果画出来。

执行这个单元格后，你应该能看到一张图，图中的红点和绿点被一条直线很好地区分开了。

code
Python
# 单元格 5: 运行并查看可视化结果
dataArr, labelMat = loadDataSet()
weights = gradAscent(dataArr, labelMat)
plotBestFit(weights)
第二部分: 实战马疝病数据预测
讲解:
理解了基本原理后，我们现在来解决真正的挑战。马疝病数据集有 21 个特征，数据量也更大。如果还用批量梯度上升，每次迭代都要计算所有样本，会非常慢。因此，我们需要一个更高效的算法：随机梯度上升 (Stochastic Gradient Ascent)。

单元格 6: 定义改进的随机梯度上升算法
讲解:
stocGradAscent1 函数实现了改进版的随机梯度上升。它和批量梯度上升最大的不同在于：每次只用一个样本来更新权重，而不是整个数据集。

这个版本做了两个关键改进：

动态学习率: 学习率 alpha 不再是一个固定值，它会随着迭代的进行而逐渐变小。这有助于算法在早期快速学习，在后期精细调整，从而更容易收敛到最佳值。

随机样本选择: 在每次大的迭代中，它会随机地从数据集中选择样本进行更新，而不是按固定顺序。这可以避免因数据排列顺序导致的潜在周期性波动，让模型训练更稳定。

code
Python
# 单元格 6: 定义改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """改进的随机梯度上升算法"""
    dataMatrix = np.array(dataMatrix)
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 1. 动态学习率
            alpha = 4 / (1.0 + j + i) + 0.0001
            # 2. 随机选取样本
            rand_idx_pos = int(np.random.uniform(0, len(dataIndex)))
            rand_idx_val = dataIndex[rand_idx_pos]
            
            # 计算并更新
            h = sigmoid(np.sum(dataMatrix[rand_idx_val] * weights))
            error = classLabels[rand_idx_val] - h
            weights = weights + alpha * error * dataMatrix[rand_idx_val]
            
            # 删除已用过的索引
            del(dataIndex[rand_idx_pos])
            
    return weights
单元格 7: 定义分类与测试函数
讲解:
为了评估我们的模型，我们还需要几个函数：

classifyVector(inX, weights): 这个函数接收一个样本的特征 inX 和训练好的权重 weights，然后输出预测的类别（0 或 1）。

colicTest(): 这是单次测试的核心流程。它会：
a. 读取训练集 horseColicTraining.txt 和测试集 horseColicTest.txt。
b. 使用 stocGradAscent1 在训练集上训练模型，得到权重。
c. 遍历测试集中的每一个样本，用 classifyVector 进行预测。
d. 比较预测结果和真实标签，统计错误数量，并计算出错误率。

multiTest(): 由于随机梯度上升的结果每次可能略有不同，为了得到一个更可靠的评估，这个函数会多次调用 colicTest()（例如 10 次），然后计算这 10 次测试的平均错误率。

code
Python
# 单元格 7: 定义分类与测试函数
def classifyVector(inX, weights):
    """根据权重对输入向量进行分类"""
    prob = sigmoid(np.sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

def colicTest():
    """单次训练和测试马疝病数据"""
    trainingSet, trainingLabels = [], []
    with open('horseColicTraining.txt') as frTrain:
        for line in frTrain.readlines():
            currLine = line.strip().split()
            lineArr = [float(currLine[i]) for i in range(21)]
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))

    # 训练模型，迭代1000次以获得更好的效果
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 1000)
    
    errorCount = 0
    numTestVec = 0
    with open('horseColicTest.txt') as frTest:
        for line in frTest.readlines():
            numTestVec += 1
            currLine = line.strip().split()
            lineArr = [float(currLine[i]) for i in range(21)]
            prediction = classifyVector(np.array(lineArr), trainWeights)
            if int(prediction) != int(currLine[21]):
                errorCount += 1
                
    errorRate = float(errorCount) / numTestVec
    print(f"本次测试的错误率是: {errorRate:.4f}")
    return errorRate

def multiTest():
    """多次测试并计算平均错误率"""
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
        
    avgErrorRate = errorSum / float(numTests)
    print(f"\n在 {numTests} 次迭代后, 平均错误率是: {avgErrorRate:.4f}")
单元格 8: 运行最终预测实验
讲解:
最后一步！执行下面的单元格来启动整个马疝病预测流程。程序会运行 10 次独立的训练和测试，并打印出每一次的错误率，最后给出一个总的平均错误率。这个平均错误率就是我们模型最终的性能评估指标。

通常，这个模型的平均错误率在 30% 到 35% 之间。

code
Python
# 单元格 8: 运行最终预测实验
multiTest()