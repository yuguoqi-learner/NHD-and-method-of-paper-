#!/home/iqr/anaconda3/envs/pytorch=3.7/bin/python
import torch
import numpy as np
import rospy
from biotac_sensors.msg import BioTacHand
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from test import test
import time
import threading
# import ThreadClass

list2 = []
data = []
index_list = []
ratiolist = []
io = 1
index = 0
def callback(msg):
    global io
    global index
    right = 0

    list1 = []

    right_syntouch= msg.bt_data[0]#右手        左手bt[1]压力290,右手[2]压力230
    left_syntouch = msg.bt_data[1]#zuoshou
    # print(msg.bt_data)
    right_pdc_data,right_pac_data = right_syntouch.pdc_data,list(right_syntouch.pac_data)
    left_pdc_data,left_pac_data = left_syntouch.pdc_data,list(left_syntouch.pac_data)
    right_data = [right_pdc_data] + right_pac_data
    left_data = [left_pdc_data] + left_pac_data
    list1.append(right_data)
    list1.append(left_data)
    list2.append(list1)
    # left_data = torch.tensor(left_data)
    # right_data = torch.tensor(right_data)

    # twofinger_data  = torch.stack((left_data,right_data),dim=0)
    # list.append(twofinger_data)

    # judge = threading.Timer(3,Judge, args=(index,list2))
    while io == 1:
        if len(list2)>= 2:
            ratioss = np.array(list2[-1])/np.array(list2[-2])

            # print(ratioss[0,2])
            # print(np.array(list2).shape)
            if   ratioss[0,2]>1.1 or ratioss[0,2]<0.9:
                index = get_result(list2)  # 等了3s

                # print('index',index)
                judge = threading.Timer(3, Judge, args=(index, list2))
                judge.start()

                io = 0
            else:
                # print('a')
                break

            # for ratios in ratioss:
            #     for ratio in ratios:
            #         if  not ratio>1.1 or ratio<0.9:
            #             index= get_result(list2)#等了3s
            #             io = 0

        else:
            # print('b')
            # index = 0
            break
def Judge(index,list2):
    # print('index is',index)
    # print('2',np.array(list2).shape)
    data = list2[index - 50:index + 250]
    np.savez('/home/iqr/yu//robot_testdata/2', data)
    data = torch.FloatTensor(data)
    data = data.transpose(1, 0)
    data = torch.unsqueeze(data, dim=0)
    cla = test(data)



    # print(data.size())

    # plt.plot(data[:,0,:])
    # plt.plot(data[:,1,:])
    # plt.show()



    # while index != 0:
    #     data = list2[index - 50:index + 250]
    #     data = torch.tensor(data)
    #     # data = data.transpose(1, 0)
    #     print(data.size())
    #     # plt.plot(data[])
    #     # plt.show()
    #     # data.unsqueeze(0)
    #     # print(data.size())
    #     # cla = test(data)
    #     break

def func(list):
    index = list.index(list[-1])
    # index_list.append(index)
    return index

class MyThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        # self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

def get_result(arg):
    threads = []  # 定义一个线程池
    t = MyThread(func, (arg,))
    # t1 = threading.Thread(target=one)  # 建立一个线程并且赋给t1，这个线程指定调用方法one，并且不带参数
    # threads.append(t)  # 把t线程装到threads线程池里

    # t2 = threading.Thread(target=two, args=(a,))

    # threads.append(t1)
    t.start()

    index = t.get_result()
    # print('index is', index)
    # time.sleep(3)
    # t1.start()
    return index


rospy.init_node('sub')
sub = rospy.Subscriber('biotac_pub',BioTacHand,callback)
rospy.spin()

