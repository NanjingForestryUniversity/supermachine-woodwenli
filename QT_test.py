# -*- codeing = utf-8 -*-
# Time : 2022/9/17 15:05
# @Auther : zhouchao
# @File: QT_test.py
# @Software:PyCharm
import logging
import socket
import numpy as np
import cv2



def rec_socket(recv_sock: socket.socket, cmd_type: str, ack: bool) -> bool:  # 未使用
    if ack:
        cmd = 'A' + cmd_type
    else:
        cmd = 'D' + cmd_type
    while True:
        try:
            temp = recv_sock.recv(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return False
        except TimeoutError as e:
            logging.error(f'超时了，错误代码: \n{e}')
            return False
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return False
        if temp == b'\xaa':
            break

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        try:
            temp += recv_sock.recv(1)
        except Exception as e:
            logging.error(f'接收报文长度失败, 错误代码: \n{e}')
            return False
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}, \n报文内容\n{temp}')
        return False

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        try:
            temp += recv_sock.recv(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}，\n报文内容\n{temp}')
            return False
    data = temp
    if cmd.strip().upper() != data[:4].decode('ascii').strip().upper():
        logging.error(f'客户端接收指令错误,\n指令内容\n{data}')
        return False
    else:
        if cmd == 'DIM':
            print(data)

        # 进行数据校验
        temp = b''
        while len(temp) < 3:
            try:
                temp += recv_sock.recv(1)
            except Exception as e:
                logging.error(f'接收报文校验失败, 错误代码: \n{e}')
                return False
        if temp == b'\xff\xff\xbb':
            return True
        else:
            logging.error(f"接收了一个完美的只错了校验位的报文，\n data: {data}")
            return False


def main():              # 在一个端口上接收文件，在另一个端口上接收控制命令
    socket_receive = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # 创建一个socket对象，AF_INET是地址簇，SOCK_STREAM是socket类型，表示TCP连接
    socket_receive.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_receive.bind(('127.0.0.1', 21123))           # 127.0.0.1是本机的回环地址，意味着socket仅接受从同一台机器上发起的连接请求。21123是端口号，它是计算机上用于区分不同服务的数字标签。
    socket_receive.listen(5)                                  # 开始监听传入的连接，5是指在拒绝连接之前，操作系统可以挂起的最大连接数量
    socket_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    socket_send.bind(('127.0.0.1', 21122))
    socket_send.listen(5)
    print('等待连接')
    socket_send_1, receive_addr_1 = socket_send.accept()
    print("连接成功：", receive_addr_1)
    # socket_send_2 = socket_send_1
    socket_send_2, receive_addr_2 = socket_receive.accept()
    print("连接成功：", receive_addr_2)
    while True:
        cmd = input().strip().upper()
        if cmd == 'IM':
            # img = cv2.imread(r"/Users/zhouchao/Library/CloudStorage/OneDrive-macrosolid/PycharmProjects/wood_color/data/data20220919/dark/rgb60.png")
            img = cv2.imread(r"D:\Projects\PycharmProjects\xiangmu_wenli\data\wenli\Huawen\huaweisecha (132).png")   # 读取图片，返回的img对象是一个NumPy数组，包含图像的像素数据。
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)               # 将BGR格式的图像转换为RGB格式
            img = np.asarray(img, dtype=np.uint8)                    # 通过np.asarray()确保图像数据是NumPy数组格式，dtype=np.uint8表示使用8位无符号整数格式存储每个颜色通道，这是图像处理中常用的数据类型。
            height = img.shape[0]                                     # 获取图像的高度
            width = img.shape[1]                                    # 获取图像的宽度
            img_bytes = img.tobytes()                                # 将图像数据转换为字节流，以便通过网络传输
            length = len(img_bytes) + 8                              # 计算报文长度，包括命令部分、宽度、高度和图像数据，以及结束符。   + 8：这个加法操作包括额外的协议或消息格式所需的字节长度 4 字节用于表示命令类型（例如 'IM'）。在某些实现中可能已经固定包含在消息的开始部分。2 字节用于图像的宽度。2 字节用于图像的高度。
            length = length.to_bytes(4, byteorder='big')       # 将报文长度转换为4字节的大端字节序
            height = height.to_bytes(2, byteorder='big')              # 将图像高度转换为2字节的大端字节序，这样可以确保图像的宽度和高度在网络传输中的顺序是正确的。
            width = width.to_bytes(2, byteorder='big')                # 将图像宽度转换为2字节的大端字节序
            send_message = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + height + width + img_bytes + b'\xff\xff\xbb' # 消息以'\xaa'开始，包含消息长度、命令代码、图像宽度和高度、图像数据本身，以及结束符'\xff\xff\xbb'来标记消息结束。
            socket_send_1.send(send_message)
            print('发送成功')
            result = socket_send_2.recv(1)
            print(result)
            # if rec_socket(socket_send_2, cmd_type=cmd, ack=True):
            #     print('接收指令成功')
            # else:
            #     print('接收指令失败')
            # if rec_socket(socket_send_2, cmd_type=cmd, ack=False):
            #     print('指令执行完毕')
            # else:
            #     print('指令执行失败')
        elif cmd == 'TR':
            # model = "/Users/zhouchao/Library/CloudStorage/OneDrive-macrosolid/PycharmProjects/wood_color/data/data20220919"
            model = r"D:\Projects\PycharmProjects\xiangmu_wenli_2\data\xiangmu_photos_wenli"   # 数据路径
            model = model.encode('ascii')           # 将字符串转换为字节流
            length = len(model) + 4                # 计算报文长度  + 4：这个加法操作通常包括额外的协议或消息格式所需的字节长度，特别是：4 字节用于存储整个消息长度的数值本身，表示消息的起始部分。
            length = length.to_bytes(4, byteorder='big')    # 将报文长度转换为4字节的大端字节序
            send_message = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + model + b'\xff\xff\xbb'
            socket_send_1.send(send_message)
            print('发送成功')
            result = socket_send_2.recv(1)
            print(result)
            # if rec_socket(socket_send_2, cmd_type=cmd, ack=True):
            #     print('接收指令成功')
            # else:
            #     print('接收指令失败')
            # if rec_socket(socket_send_2, cmd_type=cmd, ack=False):
            #     print('指令执行完毕')
            # else:
            #     print('指令执行失败')
        elif cmd == 'MD':
            # model = "/Users/zhouchao/Library/CloudStorage/OneDrive-macrosolid/PycharmProjects/wood_color/models/model_2020-11-08_20-49.p"
            # model = r"C:\Users\FEIJINTI\OneDrive\PycharmProjects\wood_color\models\model_2023-03-27_16-32.p"
            model = r"D:\Projects\PycharmProjects\xiangmu_wenli_2\models\model_2024-05-07_13-58.p"  # 模型路径
            model = model.encode('ascii')    # 将字符串转换为字节流
            length = len(model) + 4
            length = length.to_bytes(4, byteorder='big')
            send_message = b'\xaa' + length + ('  ' + cmd).upper().encode('ascii') + model + b'\xff\xff\xbb'
            socket_send_1.send(send_message)
            print('发送成功')
            result = socket_send_2.recv(1)
            print(result)
            # if rec_socket(socket_send_2, cmd_type=cmd, ack=True):
            #     print('接收指令成功')
            # else:
            #     print('接收指令失败')
            # if rec_socket(socket_send_2, cmd_type=cmd, ack=False):
            #     print('指令执行完毕')
            # else:
            #     print('指令执行失败')

if __name__ == '__main__':
    main()

