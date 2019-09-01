from numpy import *
def plotROC(predStrengths, classLabels):
    # predStrength  numpy数组或行向量组成的矩阵，通过sign函数得到
    # classLabels 是数据集的类别标签
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0) # 保留绘制光标的位置
    ySum = 0.0   #　计算ＡＵＣ的值，累加每点的高度
    numPosClas = sum(array(classLabels) == 1.0)   # 计算正例的数目
    yStep = 1 / float(numPosClas)   # y轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()   # 得到矩阵中每个元素的排序索引（从小到大）
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]: # 将矩阵转化为列表
        if classLabels[index] == 1.0:  # 每label为1.0的数据，沿y轴下降一个步长，降低真阳率
            delX = 0
            delY = yStep
        else:  # 降低假阴率
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC curve for adaboost')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the area under the curve is:', ySum * xStep)