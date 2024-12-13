import numpy as np
import torch
import matplotlib.pyplot as plt
from openTSNE import TSNE


class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """
        normalize 是否设元素为百分比形式
        """
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, labels, predicts):
        """
        :param labels:   一维标签向量 eg array([0,5,0,6,2,...],dtype=int64)
        :param predicts: 一维预测向量 eg array([0,5,1,6,3,...],dtype=int64)        
        :return:
        """
        mask = (labels != -1)
        label = labels[mask]
        predicts = predicts[mask]

        for predict, label in zip(labels, predicts):
            self.matrix[predict, label] += 1

    def getMatrix(self, normalize=True):
        """
        根据传入的normalize判断要进行percent的转换 
        如果normalize为True 则矩阵元素转换为百分比形式 
        如果normalize为False 则矩阵元素就为数量
        Returns:返回一个以百分比或者数量为元素的矩阵

        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # 计算每行的和 用于百分比计算

            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])   # 百分比转换
            # self.matrix = np.around(self.matrix, 1)   # 保留2位小数点
            self.matrix = np.around(self.matrix, 6)  # 保留6位小数点
            self.matrix[np.isnan(self.matrix)] = 0  # 可能存在NaN 将其设为0
        return self.matrix

    def genConfusionMatrix(numClass, imgPredict, imgLabel):
        mask = (imgLabel != -1)
        label = numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=numClass ** 2)
        confusionMatrix = count.reshape(numClass, numClass)
        return confusionMatrix

    def drawMatrix(self, path, acc, name):
        self.matrix = self.getMatrix(self.normalize)
        plt.figure()
        plt.clf()
        plt.imshow(self.matrix, cmap=plt.cm.GnBu)  # 仅画出颜色格子 没有值
        title_1 = 'ConfusionMatrix_' + name
        title_2 = "mean_F1:" + str(acc)
        plt.title(title_1 + '\n' + '\n' + title_2, fontsize=8)   # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)  # y轴标签
        plt.xticks(range(self.num_classes), self.labels_name, rotation=45)  # x轴标签

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.1f' % self.matrix[y, x]))
                plt.text(x, y, value, fontsize=6, verticalalignment='center', horizontalalignment='center')  # 写值

        # 后缀改为 .txt 一样
        filename = path + '.csv'
        # 写文件
        np.savetxt(filename, self.matrix, fmt='%.6f', delimiter=',')

        plt.tight_layout()  # 自动调整子图参数 使之填充整个图像区域

        plt.colorbar()  # 色条
        plt.savefig(path + '.svg', bbox_inches='tight')
        plt.savefig(path + '.pdf', bbox_inches='tight')
        plt.show()


def openTSNE(feat, labels, path):
    X = feat
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    Y = TSNE().fit(X)

    plt.figure(figsize=(10.0, 7.5))
    plt.clf()
    # cmaps_self = cm.get_cmap('gist_rainbow', 24)

    color = ['#B8CAF5', '#2FDAC2', '#1F6D04', '#A4E770', '#4DD000', '#40CC00', '#6C6E01', '#A2A000', '#F8EB0B',
             '#6EAAF4', '#0459DF', '#05246E', '#7688EE', '#05A480', '#6E0102', '#F57B7A', '#F4B8B9', '#F7B6E0',
             '#F202C0', '#DD07A4', '#A2057F', '#6E0049', '#F56AD8', '#9C9C9C']
    # color = ['#B8CAF5', '#2FDAC2', '#1F6D04', '#6C6E01', '#A2A000', '#6EAAF4', '#0459DF', '#05246E', '#6E0102',
    #          '#F57B7A', '#F4B8B9', '#A2057F', '#6E0049', '#F56AD8']
    # color = ["#B8CAF5", "#2FDAC2", "#00FF00", "#00FFFF", "#0000FF", "#FF6347", "#006400", "#6B8E23", "#008B8B",
    #          "#B0C4DE", "#7B68EE", "#E6E6FA", "#ff00ff", "#990000", "#999900", "#009900", "#009999", "#FFA07A",
    #          "#DDA0DD", "#9370DB", "#800080", "#FFB6C1", "#DB7093", "#F4A460"]
    # cmaps_WHU = colors.ListedColormap(color)

    # list_legend = []
    for i in list(set(labels)):
        x = Y[labels == i, 0]
        y = Y[labels == i, 1]
        plt.scatter(x, y, s=5, c=color[i], alpha=0.8)

    # list_legend = [0, 2, 9, 10, 14, 17, 18, 22, 23]
    # for i in list_legend:
    #     x = Y[labels == i, 0]
    #     y = Y[labels == i, 1]
    #     plt.scatter(x, y, s=5, c=color[i], alpha=0.8)

    # for i in list(set(labels)):
    #     centers_x = centers[i, 0]
    #     centers_y = centers[i, 1]
    #     pyplot.scatter(centers_x, centers_y, s=120, marker='*', c=color[i], alpha=1, linewidths=0.8, edgecolors='k')

    # pyplot.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
    #                '18', '19', '20', '21', '22', '23'], loc='upper right')
    num1 = 1.01
    num2 = 0
    num3 = 3
    num4 = 0
    # pyplot.legend(['0', '1', '2', '6', '7', '9', '10', '11', '14', '15', '16', '20', '21', '22'],
    #               bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
    # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
    #                '18', '19', '20', '21', '22', '23'],
    #               bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)

    plt.savefig(path + '.svg')
    plt.show()


# y0 = [255, 255, 255]
# y1 = [184, 202, 245]
# y2 = [47, 218, 194]
# y3 = [31, 109, 4]
# y4 = [164, 231, 112]
# y5 = [77, 208, 0]
# y6 = [65, 204, 0]
# y7 = [108, 110, 1]
# y8 = [162, 160, 0]
# y9 = [248, 235, 11]
# y10 = [110, 170, 244]
# y11 = [4, 89, 223]
# y12 = [5, 36, 110]
# y13 = [118, 136, 238]
# y14 = [5, 164, 128]
# y15 = [110, 1, 2]
# y16 = [245, 123, 122]
# y17 = [244, 184, 185]
# y18 = [247, 182, 224]
# y19 = [242, 2, 192]
# y20 = [221, 7, 164]
# y21 = [162, 5, 127]
# y22 = [110, 0, 73]
# y23 = [245, 106, 216]
# y24 = [156, 156, 156]

y0 = [255, 255, 255]  # 背景
y1 = [190, 210, 255]  # Paddy field, 水稻田
y2 = [0, 255, 197]  # Dry farm, 干农场
y3 = [38, 115, 0]  # Woodland, 林地
y4 = [163, 255, 115]  # Shrubbery, 灌木丛
y5 = [76, 230, 0]  # Sparse woodland, 稀疏林地
y6 = [85, 255, 0]  # Other forest land, 其他森林土地
y7 = [115, 115, 0]  # High-covered grassland, 高覆盖草地
y8 = [168, 168, 0]  # Medium-covered grassland, 中覆盖草地
y9 = [255, 255, 0]  # Low-covered grassland, 低覆盖草地
y10 = [115, 178, 255]  # River canal, 河道
y11 = [0, 92, 230]  # Lake, 湖泊
y12 = [0, 38, 115]  # Reservoir pond, 水库池塘
y13 = [122, 142, 245]  # Beach land, 滩涂
y14 = [0, 168, 132]  # Shoal, 沙洲
y15 = [115, 0, 0]  # Urban built-up, 城市建筑
y16 = [255, 127, 127]  # Rural settlement, 农村定居点
y17 = [255, 190, 190]  # Other construction land, 其他建设用地
y18 = [255, 190, 232]  # Sand, 沙地
y19 = [255, 0, 197]  # Gobi, 戈壁
y20 = [230, 0, 169]  # Saline-alkali land, 盐碱地
y21 = [168, 0, 132]  # Marshland, 沼泽地
y22 = [115, 0, 76]  # Bare land, 裸地
y23 = [255, 115, 223]  # Bare rock, 裸岩
y24 = [161, 161, 161]  # Ocean, 海洋

[y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22, y23,
 y24] = map(lambda x: x[::-1],
            [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22,
             y23, y24])
# print(y1)
color_list = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21, y22,
              y23, y24]


def shangse(img, nums, label):
    img_color = np.zeros((img.shape[0], img.shape[1], 3))
    if img.max() < nums:
        nums = np.int8(img.max())
        # print(nums)
    for i in range(0, nums + 1):
        img_color[img == i] = color_list[i]
        # print(color_list[i])                          #对应的颜色
        # print(f"{i}: {(img == i).max()}")  # 对应的像素点数
    # print(label)
    img_color[label == 0] = color_list[0]  # 背景
    # print([label == 0])  # 背景矩阵
    return img_color