import math
import random
import numpy as np
import matplotlib.pyplot as plt

edges = [
    [[0, 1]],
    [[1, 9], [1, 3]],
    [[2, 24]],
    [[3, 4], [3, 10]],
    [[4, 22], [4, 12]],
    [[5, 25], [5, 32]],
    [[6, 0]],
    [[7, 2]],
    [[8, 24], [8, 7]],
    [[9, 8], [9, 5]],
    [[10, 11]],
    [[11, 4]],
    [[12, 15], [12, 14]],
    [[13, 27], [13, 30]],
    [[14, 20], [14, 17]],
    [[15, 16]],
    [[16, 14]],
    [[17, 19]],
    [[18, 29], [18, 49]],
    [[19, 20]],
    [[20, 18], [20, 21]],
    [[21, 12], [21, 22]],
    [[22, 13], [22, 23]],
    [[23, 9], [23, 3]],
    [[24, 6], [24, 1]],
    [[25, 31]],
    [[26, 25], [26, 32]],
    [[27, 18], [27, 21]],
    [[28, 35]],
    [[29, 39], [29, 37]],
    [[30, 28]],
    [[31, 43], [31, 52]],
    [[32, 23], [32, 28]],
    [[33, 34]],
    [[34, 35]],
    [[35, 47], [35, 44]],
    [[36, 46]],
    [[37, 38]],
    [[38, 39]],
    [[39, 36], [39, 40]],
    [[40, 44], [40, 47]],
    [[41, 42]],
    [[42, 48]],
    [[43, 35], [43, 26]],
    [[44, 29], [44, 49]],
    [[45, 50]],
    [[46, 40]],
    [[47, 51], [47, 45]],
    [[48, 8]],
    [[49, 27], [49, 30]],
    [[50, 51]],
    [[51, 43], [51, 52]],
    [[52, 48], [52, 41]]
]
pos = [(-46.3, 16.4), (-32.3, 8.9), (-32.2, 31.6), (-32.2, 0.6), (-32.2, -2), (-14.8, 12.5), (-43.2, 16.8),
       (-29.4, 28.9), (-29.6, 11), (-29.4, 8.8),
       (-47.3, 0.5), (-49.2, -2.1), (-32.3, -24.5), (-23.1, -2.1), (-32.3, -27.1), (-45.2, -24.7), (-42.8, -27.3),
       (-31.3, -37.8), (-8.0, -27.3), (-29.6, -37.8),
       (-29.3, -27.2), (-29.5, -24.8), (-29.5, -2.3), (-29.5, 0.5), (-32.3, 11.1), (-12, 13), (-6.42, 0.3),
       (-9.36, -24.7), (-6.4, -2.1), (19.2, -27.2),
       (-20.5, -2.5), (9.1, 18.5), (-9.25, 0.3), (4.3, -1.9), (6.5, -1.9), (19.2, -2.2), (34.4, -27.2), (19.1, -43),
       (21.9, -45.9), (21.8, -27.3),
       (21.8, -24.7), (44.1, 18.5), (41, 20.6), (19.4, 0.5), (19.2, -24.6), (39.8, -2.1), (37, -24.8), (21.6, -2),
       (9.4, 20.5), (-7.1, -24.7),
       (36.6, 0.5), (21.8, 0.6), (11.6, 18.5)
       ]


class PathCreator:
    def __init__(self, carSpeed, runTime, timeSlot, pathNum, forwardProbability, random_seed):
        self.carSpeed = carSpeed
        self.timeSlot = timeSlot
        self.pathNum = pathNum  # path的数量
        self.path = []  # 记录访问node的顺序
        self.pathPoint = []  # 记录具体的路径节点
        self.runTime = runTime
        self.moveLength = self.timeSlot * self.carSpeed
        self.stepLength = self.moveLength
        self.forwardProbability = forwardProbability
        random.seed(random_seed)

    def initializeStartPosition(self):

        # temp=random.choice([39,40,44,29])
        temp = 39
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([51,47,35,43])
        temp = 51
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([52,48,31])
        temp = 52
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([49,27,18])
        temp = 49
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([28,26,32])
        temp = 28
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([25,5])
        temp = 25
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([30,13])
        temp = 30
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([20,21,14,12])
        temp = 20
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([22,4,3,23])
        temp = 22
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # temp=random.choice([9,8,1,24])
        temp = 9
        self.path.append([temp])
        self.pathPoint.append([np.array([pos[temp][0], 0, pos[temp][1]])])

        # print(self.pathPoint)

    def dis(self, p1, p2):
        sum = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        return math.sqrt(sum)

    def norm(self, v):
        return math.sqrt(sum(e ** 2 for e in v))

    def normalized(self, v):
        len = self.norm(v)
        if len == 0:
            return [0, 0]
        return [x / len for x in v]

    def getAngle(self, v1, v2):
        '''计算夹角，并返回一个度数'''
        v1 = np.array(v1)
        v2 = np.array(v2)
        l1 = np.sqrt(v1.dot(v1))
        l2 = np.sqrt(v2.dot(v2))
        dian = v1.dot(v2)
        cos_ = dian / (l1 * l2)
        angle = np.arccos(cos_)
        return np.rad2deg(angle)

    def createPath(self):
        self.pathPoint = []
        self.path = []
        self.initializeStartPosition()
        for i in range(self.pathNum):
            currentPoint = [self.pathPoint[i][0][0], self.pathPoint[i][0][2]]

            curentNode = self.path[i][0]
            preNode = curentNode
            selectPoint = True

            for j in range(1, int(self.runTime / self.timeSlot)):
                # 查询路径当前点可走的方向
                # 如果只有一个方向
                if len(edges[curentNode]) == 1:
                    direction = [pos[edges[curentNode][0][1]][0] - currentPoint[0],
                                 pos[edges[curentNode][0][1]][1] - currentPoint[1]]
                    # 如果从当前点到下一个点的距离小于了，每步走的距离，则需要继续选定方向，反之则直接行走
                    if self.norm(direction) > self.stepLength:
                        '''距离大于了单步距离'''
                        nextPoint = [ii * self.stepLength for ii in self.normalized(direction)]
                        nextPoint = [nextPoint[0] + currentPoint[0], nextPoint[1] + currentPoint[1]]
                        self.pathPoint[i].append(np.array([nextPoint[0], 0, nextPoint[1]]))
                        currentPoint = nextPoint
                        # 重置单步需要移动的距离
                        self.stepLength = self.moveLength
                    else:
                        '''距离小于了单步的距离,则更新当前点到下一个node 的位置'''
                        self.stepLength = self.moveLength - self.norm(direction)
                        # 更新当前节点
                        preNode = curentNode
                        curentNode = edges[curentNode][0][1]
                        # 将节点加入访问顺序中去
                        self.path[i].append(curentNode)
                        # 更新当前节点
                        currentPoint = pos[curentNode]
                        if (math.isclose(self.stepLength, 0, rel_tol=0.00001)):
                            self.stepLength = self.moveLength
                            # 将节点加入到路径当中
                            self.pathPoint[i].append(np.array([pos[curentNode][0], 0, pos[curentNode][1]]))

                else:
                    '''此时当前节点有两条出度，即有两个可以行驶的方向'''
                    if selectPoint:
                        if preNode != curentNode:
                            # 首先计算行驶方向与两可选择方向的夹角
                            direction1 = [pos[edges[curentNode][0][1]][0] - currentPoint[0],
                                          pos[edges[curentNode][0][1]][1] - currentPoint[1]]
                            direction2 = [pos[edges[curentNode][1][1]][0] - currentPoint[0],
                                          pos[edges[curentNode][1][1]][1] - currentPoint[1]]
                            direction = [pos[curentNode][0] - pos[preNode][0], pos[curentNode][1] - pos[preNode][1]]
                            angle1 = self.getAngle(direction, direction1)
                            angle2 = self.getAngle(direction, direction2)
                            # 找到哪个才是直行的节点
                            forwardNode = edges[curentNode][0][1]
                            turnningNode = edges[curentNode][1][1]
                            if angle1 > angle2:
                                forwardNode = edges[curentNode][1][1]
                                turnningNode = edges[curentNode][0][1]
                            chooseNode = forwardNode
                            # print(f"direction {direction},angle1 {angle1},angle2 {angle2}")
                            if (random.random() < self.forwardProbability):
                                pass
                            else:
                                chooseNode = turnningNode
                        else:
                            # print('*******************************************')
                            '''根据当前的出度，随机选择一个作为行驶方向'''
                            tRange = [x for x in range(len(edges[curentNode]))]
                            chooseNode = edges[curentNode][random.choice(tRange)][1]
                        # print(f"curentNode:{curentNode},chooseNode:{chooseNode} dis:{self.dis(pos[curentNode],pos[chooseNode])}")

                        selectPoint = False
                        # 截至这里方向点选取完毕
                    direction = [pos[chooseNode][0] - currentPoint[0], pos[chooseNode][1] - currentPoint[1]]
                    # 如果从当前点到下一个点的距离小于了，每步走的距离，则需要继续选定方向，反之则直接行走
                    if (self.norm(direction) > self.stepLength):
                        '''距离大于了单步距离'''
                        nextPoint = [ii * self.stepLength for ii in self.normalized(direction)]

                        nextPoint = [nextPoint[0] + currentPoint[0], nextPoint[1] + currentPoint[1]]

                        # print(f"i:{i},stepLength:{self.stepLength},self.normalized(direction){self.normalized(direction)},nextPoint:{nextPoint},Dis:{self.dis(nextPoint,pos[curentNode])}")
                        self.pathPoint[i].append(np.array([nextPoint[0], 0, nextPoint[1]]))
                        currentPoint = nextPoint
                        # 重置单步需要移动的距离
                        self.stepLength = self.moveLength
                        # print("*************************************************************************")
                    else:
                        '''距离小于了单步的距离,则更新当前点到下一个node 的位置'''
                        self.stepLength = self.moveLength - self.norm(direction)
                        # 更新当前节点
                        preNode = curentNode
                        curentNode = chooseNode
                        # 将节点加入访问顺序中去
                        self.path[i].append(curentNode)
                        # 更新当前节点
                        currentPoint = pos[curentNode]
                        selectPoint = True
                        if (math.isclose(self.stepLength, 0, rel_tol=0.00001)):
                            self.stepLength = self.moveLength
                            # 将节点加入到路径当中
                            self.pathPoint[i].append(np.array([pos[curentNode][0], 0, pos[curentNode][1]]))


# pathcreator = PathCreator(0.8, 4000, 0.2, 10, 0.5,1)
# pathcreator.createPath()
#
# X = [x[0] for x in pathcreator.pathPoint[9]]
# Y = [x[2] for x in pathcreator.pathPoint[9]]
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.scatter(X, Y)
# plt.show()
