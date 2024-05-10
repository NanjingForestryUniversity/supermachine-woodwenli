# -*- coding: UTF-8 -*-
# @Time : 2024/4/15 19:03
# @Auther : DUANMU
# @File : wenli_classifer.py
# @Software : PyCharm
import logging
import numpy as np
import cv2
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from skimage.filters import sobel
from skimage import filters
from skimage.morphology import opening, square
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.stats import kurtosis, skew
from skimage import transform, util
from skimage.feature import hog
from skimage import color, exposure
import lightgbm as lgb
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
import pickle
import time

from root_dir import ROOT_DIR
import utils

class TextureClass:
    def __init__(self, load_from=None, w=4096, h=1200, debug_mode=False):
        """
        初始化纹理分类器
        :param load_from:
        :param w:
        :param h:
        :param debug_mode:
        """
        if load_from is None:
            if w is None or h is None:
                print("It will damage your performance if you don't set w and h")
                raise ValueError("w or h is None")
            self.w, self.h= w, h
            # self.model = RandomForestClassifier(n_estimators=100)
            # self.model = lgb.LGBMClassifier(
            #     n_estimators=100,
            #     max_depth=5,
            #     num_leaves=16,  # 减小叶节点数
            #     min_data_in_leaf=40,  # 增加每个叶子的最小数据量
            #     lambda_l1=0.1,  # 增强L1正则化
            #     lambda_l2=0.1,  # 增强L2正则化
            #     boosting_type='gbdt',  # 使用传统的梯度提升决策树
            #     objective='binary',  # 二分类问题
            #     learning_rate=0.05,  # 可以尝试降低学习速率
            #     subsample=0.8,  # 子样本比率
            #     colsample_bytree=0.8,  # 基于树的列采样
            #     metric='binary_logloss',  # 评价指标
            #     verbose=-1  # 不输出任何东西，包括警告
            # )
            self.model = lgb.LGBMClassifier(n_estimators=100 ,verbose=-1)
            # self.model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10,min_samples_leaf=2,
            #                                     min_samples_split=4,max_features='sqrt')
            # self.model = AdaBoostClassifier()
            # self.model = GradientBoostingClassifier()
            # self.model = ExtraTreesClassifier()
            # self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        else:
            self.load(load_from)
        self.debug_mode = debug_mode
        self.log = utils.Logger(is_to_file=debug_mode)
        self.image_num = 0

    def extract_features(self, img):
        # """使用局部二值模式（LBP）提取图像的纹理特征"""
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # if gray.dtype != np.uint8:
        #     gray = (gray * 255).astype(np.uint8)
        #
        # # 设置LBP参数
        # radius = 1  # LBP算法中圆的半径
        # n_points = 8 * radius  # 统一模式用的点数
        # lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        #
        # # 计算LBP的直方图
        # n_bins = int(lbp.max() + 1)  # 加1因为直方图的区间是开区间
        # hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        # festures = hist
        #
        # return festures

        """使用局部二值模式（LBP）提取图像的纹理特征，优化版"""
        # 图像下采样
        img_resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # gray = self.preprocess_image(img_resized)
        if gray.dtype != np.uint8:
            gray = (gray * 255).astype(np.uint8)

        # 设置LBP参数
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

        # 计算LBP的直方图
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        features = hist
        return features

    def augment_image(self, img):
        """对输入图像应用随机数据增强"""
        # 随机旋转 ±30 度
        angle = np.random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # 随机水平翻转
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)

        # 随机调整亮度 ±50
        value = int(np.random.uniform(-50, 50))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, value)
        v[v > 255] = 255
        v[v < 0] = 0
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return img

    def get_image_data(self, img_dir="./data/wenli/Huawen",augment=False):
        """
        :param img_dir: 图像文件的路径
        :return: 图像数据
        """
        img_data = []
        img_name = []
        utils.mkdir_if_not_exist(img_dir)
        files = os.listdir(img_dir)
        if len(files) == 0:
            return False
        for file in files:
            path = os.path.join(img_dir, file)
            if self.debug_mode:
                self.log.log(path)
            train_img = cv2.imread(path)
            if augment:
                train_img = self.augment_image(train_img)  # 应用数据增强
            data = self.extract_features(train_img)
            img_data.append(data)
            img_name.append(file)
        img_data = np.array(img_data)

        return img_data, img_name

    def get_train_data(self, data_dir=None, plot_2d=False, save_data=False, augment=False):
        """
        获取图像数据
        :return: x_data, y_data
        """
        print("开始加载训练数据...")
        data_dir = os.path.join(ROOT_DIR, "data", "wenli") if data_dir is None else data_dir
        hw_data, hw_name = self.get_image_data(img_dir=os.path.join(data_dir, "Huawen"), augment=augment)
        zw_data, zw_name = self.get_image_data(img_dir=os.path.join(data_dir, "Zhiwen"), augment=augment)
        if (hw_data is False) or (zw_data is False) :
            return False
        x_data = np.vstack((hw_data, zw_data))

        hw_label = np.zeros(len(hw_data)).T   # 为"Huawen"类图像赋值标签0
        zw_label = np.ones(len(zw_data)).T   # 为"Zhiwen"类图像赋值标签1
        y_data = np.hstack((hw_label, zw_label))

        img_name = hw_name + zw_name

        # if plot_2d:
        #     plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap='viridis')
        #     plt.show()
        if save_data:
            with open(os.path.join("data", "data.p"), "rb") as f:
                pass
        if (hw_data is False) or (zw_data is False):
            print("未找到有效的训练数据")
            return False
        print("训练数据加载完成")
        return x_data, y_data, img_name

    def fit_pictures(self, data_path=ROOT_DIR, file_name=None, augment=False):
        """
        根据给出的data_path 进行 fit.如果没有给出data目录，那么将会使用当前文件夹
        :param data_path:
        :return:
        """
        print("开始训练模型...")
        # 训练数据文件位置
        result = self.get_train_data(data_path, plot_2d=True, augment=augment)
        if result is False:
            print("训练数据加载失败，中止训练")
            return 0
        x, y, name = result
        print("训练数据加载成功，开始模型训练")
        score = self.fit(x, y)
        print('model score', score)
        model_name = self.save(file_name)
        return model_name

    def fit(self, X, y):
        """训练模型"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        print(confusion_matrix(y_test, y_pred))

        pre_score = accuracy_score(y_test, y_pred)  # 计算测试集的准确率
        self.log.log("Test accuracy is:" + str(pre_score * 100) + "%.")

        y_pred = self.model.predict(X_train)
        pre_score = accuracy_score(y_train, y_pred)
        self.log.log("Train accuracy is:" + str(pre_score * 100) + "%.")

        y_pred = self.model.predict(X)
        pre_score = accuracy_score(y, y_pred)
        self.log.log("Total accuracy is:" + str(pre_score * 100) + "%.")

        return int(pre_score * 100)

    def predict(self, img):
        """预测图像纹理"""
        if self.debug_mode:
            cv2.imwrite(str(self.image_num) + ".bmp", img)
            self.image_num += 1
        features = self.extract_features(img)              # 提取图像特征
        features = np.array(features)  # 将列表转换为 Numpy 数组
        features = features.reshape(1, -1)  # 使用 reshape(1, -1) 将特征数组转换成适合模型预测的形状。这里的 1 表示样本数为一（单个图像），-1 表示自动计算特征数量。
        pred_wenli = self.model.predict(features)   # 使用模型预测图像的纹理
        if self.debug_mode:
            self.log.log(features)
        return int(pred_wenli[0])    # 从模型预测结果中提取第一个元素（假设预测结果是一个数组），并将其转换为整数。这通常是分类任务的类别标签。


    def save(self, file_name):
        """保存模型到文件"""
        if file_name is None:
            file_name = "model_" + time.strftime("%Y-%m-%d_%H-%M") + ".p"
        file_name = os.path.join(ROOT_DIR, "models", file_name)
        model_dic = { "model": self.model,"w": self.w, "h": self.h}
        with open(file_name, "wb") as f:
            pickle.dump(model_dic, f)
        self.log.log("Save file to '" + str(file_name) + "'")
        return file_name


    def load(self, path=None):
        if path is None:
            path = os.path.join(ROOT_DIR, "models")
            utils.mkdir_if_not_exist(path)
            model_files = os.listdir(path)
            if len(model_files) == 0:
                self.log.log("No model found!")
                return 1
            self.log.log("./ Models Found:")
            _ = [self.log.log("├--" + str(model_file)) for model_file in model_files]
            file_times = [model_file[6:-2] for model_file in model_files]
            latest_model = model_files[int(np.argmax(file_times))]
            self.log.log("└--Using the latest model: " + str(latest_model))
            path = os.path.join(ROOT_DIR, "models", str(latest_model))
        if not os.path.isabs(path):
            logging.warning('给的是相对路径')
            return -1
        if not os.path.exists(path):
            logging.warning('文件不存在')
            return -1
        with open(path, "rb") as f:
            model_dic = pickle.load(f)

            self.model = model_dic["model"]
            self.w = model_dic["w"]
            self.h = model_dic["h"]

        return 0

if __name__ == '__main__':
    from config import Config

    # 加载配置设置
    settings = Config()

    # 初始化 TextureClass 实例
    texture = TextureClass(w=4096, h=1200, debug_mode=False)
    print("初始化纹理分类器完成")

    # 获取数据路径
    data_path = settings.data_path
    # texture.correct()  # 如果有色彩校正的步骤可以添加
    # texture.load()  # 如果需要加载先前的模型可以调用

    # 训练模型并保存
    model_path = texture.fit_pictures(data_path=data_path)
    print(f"模型保存在 {model_path}")

    # # 加载K-means数据并进行数据调整
    # x_data, y_data, labels, img_names = texture.get_kmeans_data(data_path, plot_2d=True)
    # send_data = texture.data_adjustments(x_data, y_data, labels, img_names)
    # print(f"调整后的数据已发送/保存")

    # 测试单张图片的预测性能
    pic = cv2.imread(r"data/wenli/Zhiwen/rgb98.png")
    start_time = time.time()
    texture_type = texture.predict(pic)
    end_time = time.time()
    print("单次预测耗时:", (end_time - start_time) * 1000, "毫秒")
    print("预测的纹理类型:", texture_type)

    # 如果有批量处理或性能测试需求
    # 总时间 = 0
    # for i in range(100):
    #     start_time = time.time()
    #     _ = texture.predict(pic)
    #     end_time = time.time()
    #     total_time += (end_time - start_time)
    # print("平均预测时间:", (total_time / 100) * 1000, "毫秒")
