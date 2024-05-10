# -*- coding: utf-8 -*-
"""
Created on Nov 3 21:18:26 2020

@author: l.z.y
@e-mail: li.zhenye@qq.com
"""
import pathlib

file_path = pathlib.Path(__file__) # 获得当前文件路径
ROOT_DIR = file_path.parent       # 获得当前文件的父目录

# pathlib的作用是将文件路径转换为操作系统的路径格式，这样可以避免不同操作系统的路径格式不同的问题