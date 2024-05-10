# -*- codeing = utf-8 -*-
# Time : 2022/10/17 11:07
# @Auther : zhouchao
# @File: config.py
# @Software:PyCharm
import json
import os
from pathlib import WindowsPath

from root_dir import ROOT_DIR


class Config(object):
    model_path = ROOT_DIR / 'config.json'  # “/”运算符被重载，用于连接两个路径，表示 ROOT_DIR 目录下的 config.json 文件

    def __init__(self):                          # 初始化方法
        self._param_dict = {}                    # 创建一个空字典
        if os.path.exists(Config.model_path):
            self._read()                        # 如果 config.json 文件存在，调用 _read 方法
        else:
            self.model_path = str(ROOT_DIR / 'models/model_2024-04-18_10-16.p')   # 模型路径
            self.data_path = str(ROOT_DIR / 'data/xiangmu_photos_wenli')                # 数据路径
            self.database_addr = str("mysql+pymysql://root:@localhost:3306/orm_test")  # 测试用数据库地址
            self._param_dict['model_path'] = self.model_path           # 将模型路径写入 _param_dict 属性中
            self._param_dict['data_path'] = self.data_path
            self._param_dict['database_addr'] = self.database_addr

    def __setitem__(self, key, value):                          # 重载 __setitem__ 方法
        if key in self._param_dict:                             # 如果 key 在 _param_dict 中  key是键，value是值 用于设置值
            self._param_dict[key] = value                       # 将键值对写入 _param_dict 属性中
            self._write()                                       # 将 _param_dict 属性写入 config.json 文件

    def __getitem__(self, item):                                # 重载 __getitem__ 方法
        if item in self._param_dict:                            # 如果 item 在 _param_dict 中 item是键 value是值   用于获取值
            return self._param_dict[item]                       # 返回 _param_dict 中 item 键的值

    def __setattr__(self, key, value):
        self.__dict__[key] = value                                   # 直接在对象的 __dict__ 属性（这是一个存储对象所有属性的字典）中设置键（属性名）和值
        if '_param_dict' in self.__dict__ and key != '_param_dict':  # 如果 _param_dict 属性存在，且不是 _param_dict 属性本身
            if isinstance(value, WindowsPath):                       # 如果 value 是 WindowsPath 对象，将其转换为字符串
                value = str(value)                                   # WindowsPath 对象不能被 json 序列化，需要转换为字符串
            self.__dict__['_param_dict'][key] = value                # 将键值对写入 _param_dict 属性中
            self._write()                                            # 将 _param_dict 属性写入 config.json 文件

    def _read(self):                                                # 读取 config.json 文件
        with open(Config.model_path, 'r') as f:                     # 打开 config.json 文件
            self._param_dict = json.load(f)                         # 读取文件内容，将其转换为字典
            self.data_path = self._param_dict['data_path']          # 从字典中读取 data_path 键的值
            self.model_path = self._param_dict['model_path']        # 从字典中读取 model_path 键的值
            self.database_addr = self._param_dict['database_addr']

    def _write(self):                                                # 将 _param_dict 属性写入 config.json 文件
        with open(Config.model_path, 'w') as f:                      # 打开 config.json 文件
            json.dump(self._param_dict, f)                           # 将 _param_dict 写入文件


if __name__ == '__main__':
    config = Config()
    print(config.model_path)
    print(config.data_path)
