# -*- coding: utf-8 -*-
"""
Created on Nov 3 21:18:26 2020

@author: l.z.y
@e-mail: li.zhenye@qq.com
"""
import logging
import os
import shutil
import time
import socket
import numpy as np


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):                                # 如果 is_delete 为 True，函数会检查 dir_name 是否已经存在。
                shutil.rmtree(dir_name)                                 # 如果存在，它会使用 shutil.rmtree 删除整个目录结构，并输出一个信息消息。
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):   # 使用 os.path.exists 来检查 dir_name 是否存在。
            os.makedirs(dir_name)          # 如果不存在，它会使用 os.makedirs 创建整个目录树。
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


def create_file(file_name):
    """
    创建文件
    :param file_name: 文件名
    :return: None
    """
    if os.path.exists(file_name):
        print("文件存在：%s" % file_name)
        return False
        # os.remove(file_name)  # 删除已有文件
    if not os.path.exists(file_name):
        print("文件不存在，创建文件：%s" % file_name)
        open(file_name, 'a').close()
        return True


class Logger(object):          # 日志类
    def __init__(self, is_to_file=False, path=None):
        self.is_to_file = is_to_file        # 布尔值，用于决定日志是保存到文件(True)还是输出到控制台(False)
        if path is None:                    # 如果没有指定日志文件的路径，则默认为当前目录下的 wood.log
            path = "wood.log"
        self.path = path                   # 指定保存日志的文件路径。
        create_file(path)                  # 确保日志文件存在，如果不存在则创建它

    def log(self, content):
        if self.is_to_file:                   # 如果指定了日志文件的路径，则将日志信息写入到日志文件中
            with open(self.path, "a") as f:   # 以追加的方式打开日志文件,'a'（Append 模式）: 在这种模式下，如果文件已经存在，新的数据会被写入到文件的末尾。如果文件不存在，它会被创建。
                print(time.strftime("[%Y-%m-%d_%H-%M-%S]:"), file=f)  # 将当前时间戳和实际日志内容写入到文件，显示为[年-月-日_时-分-秒]: 的格式。
                print(content, file=f)
        else:
            print(content)   # 如果没有指定日志文件的路径，则将日志信息输出到控制台


def try_connect(connect_ip: str, port_number: int, is_repeat: bool = False, max_reconnect_times: int = 50) -> (
                bool, socket.socket):
    """
    尝试连接.

    :param is_repeat: 是否是重新连接
    :param max_reconnect_times:最大重连次数
    :return: (连接状态True为成功, Socket / None)
    """
    reconnect_time = 0
    while reconnect_time < max_reconnect_times:
        logging.warning(f'尝试{"重新" if is_repeat else ""}发起第{reconnect_time + 1}次连接...')
        try:
            connected_sock = PreSocket(socket.AF_INET, socket.SOCK_STREAM)
            connected_sock.connect((connect_ip, port_number))
        except Exception as e:
            reconnect_time += 1
            logging.error(f'第{reconnect_time}次连接失败... 5秒后重新连接...\n {e}')
            time.sleep(5)
            continue
        logging.warning(f'{"重新" if is_repeat else ""}连接成功')
        return True, connected_sock
    return False, None


class PreSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_pack = b''
        self.settimeout(5)

    def receive(self, *args, **kwargs):
        if self.pre_pack == b'':
            return self.recv(*args, **kwargs)
        else:
            data_len = args[0]
            required, left = self.pre_pack[:data_len], self.pre_pack[data_len:]
            self.pre_pack = left
            return required

    def set_prepack(self, pre_pack: bytes):
        temp = self.pre_pack
        self.pre_pack = temp + pre_pack


class DualSock(PreSocket):
    def __init__(self, connect_ip='127.0.0.1', recv_port: int = 21122, send_port: int = 21123):
        super().__init__()
        received_status, self.received_sock = try_connect(connect_ip=connect_ip, port_number=recv_port)   # 这两行代码分别设置接收和发送的sockets。
        send_status, self.send_sock = try_connect(connect_ip=connect_ip, port_number=send_port)
        self.status = received_status and send_status

    def send(self, *args, **kwargs) -> int:
        return self.send_sock.send(*args, **kwargs)

    def receive(self, *args, **kwargs) -> bytes:
        return self.received_sock.receive(*args, **kwargs)

    def set_prepack(self, pre_pack: bytes):
        self.received_sock.set_prepack(pre_pack)

    def reconnect(self, connect_ip='127.0.0.1', recv_port:int = 21122, send_port: int = 21123):
        received_status, self.received_sock = try_connect(connect_ip=connect_ip, port_number=recv_port)
        send_status, self.send_sock = try_connect(connect_ip=connect_ip, port_number=send_port)
        return received_status and send_status


def receive_sock(recv_sock: PreSocket, pre_pack: bytes = b'', time_out: float = -1.0, time_out_single=5e20) -> (
bytes, bytes):
    """
    从指定的socket中读取数据.

    :param recv_sock: 指定sock
    :param pre_pack: 上一包的粘包内容
    :param time_out: 每隔time_out至少要发来一次指令,否则认为出现问题进行重连，小于0则为一直等
    :param time_out_single: 单次指令超时时间，单位是秒
    :return: data, next_pack
    """
    recv_sock.set_prepack(pre_pack)
    # 开头校验
    time_start_recv = time.time()
    while True:
        if time_out > 0:
            if (time.time() - time_start_recv) > time_out:
                logging.error(f'指令接收超时')
                return b'', b''
        try:
            temp = recv_sock.receive(1)
        except ConnectionError as e:
            logging.error(f'连接出错, 错误代码:\n{e}')
            return b'', b''
        except TimeoutError as e:
            # logging.error(f'超时了，错误代码: \n{e}')
            logging.info('运行中,等待指令..')
            continue
        except socket.timeout as e:
            logging.info('运行中,等待指令..')
            continue
        except Exception as e:
            logging.error(f'遇见未知错误，错误代码: \n{e}')
            return b'', b''
        if temp == b'\xaa':
            break

    # 接收开头后，开始进行时间记录
    time_start_recv = time.time()

    # 获取报文长度
    temp = b''
    while len(temp) < 4:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文的长度不正确, 错误代码: \n{e}')
            return b'', b''
    try:
        data_len = int.from_bytes(temp, byteorder='big')
    except Exception as e:
        logging.error(f'转换失败,错误代码 \n{e}')
        return b'', b''

    # 读取报文内容
    temp = b''
    while len(temp) < data_len:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(data_len)
        except Exception as e:
            logging.error(f'接收报文内容失败, 错误代码: \n{e}')
            return b'', b''
    data, next_pack = temp[:data_len], temp[data_len:]
    recv_sock.set_prepack(next_pack)
    next_pack = b''

    # 进行数据校验
    temp = b''
    while len(temp) < 3:
        if (time.time() - time_start_recv) > time_out_single:
            logging.error(f'单次指令接收超时')
            return b'', b''
        try:
            temp += recv_sock.receive(1)
        except Exception as e:
            logging.error(f'接收报文校验失败, 错误代码: \n{e}')
            return b'', b''
    if temp == b'\xff\xff\xbb':
        return data, next_pack
    else:
        logging.error(f"接收了一个完美的只错了校验位的报文")
        return b'', b''


def parse_protocol(data: bytes) -> (str, any):    # data: 参数类型为 bytes，表示接收到的报文数据。返回类型 (str, any) 表示函数返回一个元组，其中第一个元素是指令类型的字符串，第二个元素是指令对应的内容，内容的类型可以是任何类型（由指令决定）。
    """
    指令转换.

    :param data:接收到的报文
    :return: 指令类型和内容
    """
    try:
        assert len(data) > 4
    except AssertionError:
        logging.error('指令转换失败，长度不足5')
        return '', None
    cmd, data = data[:4], data[4:]                        # 从 data 中取出前 4 个字节作为指令类型，剩下的部分作为指令内容。
    cmd = cmd.decode('ascii').strip().upper()             # 将指令类型转换为字符串，并去除首尾空格，然后转换为大写。
    if cmd == 'IM':
        n_rows, n_cols, img = data[:2], data[2:4], data[4:]               # 按照协议,先是两个字节的行数(高),两个字节的列数(宽),后面是图像数据
        try:
            n_rows, n_cols = [int.from_bytes(x, byteorder='big') for x in [n_rows, n_cols]]
        except Exception as e:
            logging.error(f'长宽转换失败, 错误代码{e}, 报文大小: n_rows:{n_rows}, n_cols: {n_cols}')
            return '', None
        try:
            assert n_rows * n_cols * 3 == len(img)
            # 因为是float32类型 所以长度要乘12 ，如果是uint8则乘3
        except AssertionError:
            logging.error('图像指令IM转换失败，数据长度错误')
            return '', None
        img = np.frombuffer(img, dtype=np.uint8).reshape((n_rows, n_cols, -1))
        return cmd, img
    elif cmd == 'TR':
        data = data.decode('ascii')
        return cmd, data
    elif cmd == 'MD':
        data = data.decode('ascii')
        return cmd, data


def ack_sock(send_sock: socket.socket, cmd_type: str) -> bool:    # 未使用
    '''
    发送应答
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :return:是否发送成功
    '''
    msg = b'\xaa\x00\x00\x00\x05' + (' A' + cmd_type).upper().encode('ascii') + b'\xff\xff\xff\xbb'
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送应答失败，错误类型：{e}')
        return False
    return True


def done_sock(send_sock: socket.socket, cmd_type: str, result = '') -> bool:  # 未使用
    '''
    发送任务完成指令
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :param result:数据
    :return:是否发送成功
    '''
    cmd_type = cmd_type.strip().upper()
    if (cmd_type == "TR") or (cmd_type == "MD"):
        if result != '':
            logging.error('结果在这种指令里很没必要')
        result = b'\xff'
    elif cmd_type == 'IM':
        if result == 0:
            result = b'H'
        elif result == 1:
            result = b'Z'
    length = len(result) + 4
    length = length.to_bytes(4, byteorder='big')
    msg = b'\xaa' +length + (' D' + cmd_type).upper().encode('ascii') + result + b'\xff\xff\xbb'
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False
    return True


def simple_sock(send_sock: socket.socket, cmd_type: str, result) -> bool:
    '''
    发送任务完成指令
    :param cmd_type:指令类型
    :param send_sock:指定sock
    :param result:数据
    :return:是否发送成功
    '''
    cmd_type = cmd_type.strip().upper()  # 去除空格并转换为大写
    if cmd_type == 'IM':
        if result == 0:
            msg = b'H'
        elif result == 1:
            msg = b'Z'
    elif cmd_type == 'TR':
        msg = b'A'
    elif cmd_type == 'MD':
        msg = b'D'
        result = result.encode('ascii')
        result = b',' + result
        length = len(result)
        msg = msg + length.to_bytes(4, 'big') + result
    try:
        send_sock.send(msg)
    except Exception as e:
        logging.error(f'发送完成指令失败，错误类型：{e}')
        return False
    return True



if __name__ == '__main__':
    log = Logger(is_to_file=True)
    log.log("nihao")
    import numpy as np

    a = np.ones((100, 100, 3))
    log.log(a.shape)
