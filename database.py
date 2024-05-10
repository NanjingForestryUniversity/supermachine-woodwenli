# -*- codeing = utf-8 -*-
# Time : 2022/10/20 13:54
# @Auther : zhouchao
# @File: database.py
# @Software:PyCharm
import datetime
import time

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import INTEGER, VARCHAR
from sqlalchemy import Column, TIMESTAMP, DATETIME
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()


class Wood(Base):
    __tablename__ = 'color'

    id = Column(INTEGER, primary_key=True)
    color = Column(VARCHAR(256), nullable=False)
    time = Column(DATETIME, nullable=False)


    def __init__(self, color):
        self.time = datetime.datetime.now()
        self.color = color


class Database(object):
    def __init__(self, database_addr):
        self.database_addr = database_addr

    def init_db(self):
        engine = create_engine(self.database_addr, encoding="utf-8", echo=True)
        Base.metadata.create_all(engine)
        print('Create table successfully!')

    def add_data(self, color):
        # 初始化数据库连接
        engine = create_engine(self.database_addr, encoding="utf-8")
        # 创建DBSession类型
        DBSession = sessionmaker(bind=engine)

        # 创建session对象
        session = DBSession()
        # 插入单条数据
        # 创建新User对象
        new_wood = Wood(color)
        # 添加到session
        session.add(new_wood)
        # 提交即保存到数据库
        session.commit()


if __name__ == '__main__':
    test_addr = "mysql+pymysql://root:@localhost:3306/color"
    database = Database(test_addr)
    # database.init_db()
    t1 = time.time()
    for i in range(100):
        database.add_data('middle')
    t2 = time.time()
    print((t2-t1)/100)