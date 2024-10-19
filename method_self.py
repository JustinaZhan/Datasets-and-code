import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import linprog


# 第一大块： 估计缺失值
# 函数功能：找到社交信任网络下的每个决策者由信任度从高到低的决策者的索引，结果中的第一个数为决策者本身
def knn(T):
    p = T.shape[0]
    indices = np.arange(0, p, 1)
    indices1 = indices.reshape(p, 1)
    knn_in = np.repeat(indices1, p, axis=1)
    T1 = T.copy()
    np.fill_diagonal(T1, -1)
    knn_in1 = np.argsort(T1, axis=1)[:, ::-1]
    knn_in[:, 1:] = knn_in1[:, :p-1]

    return knn_in


# 函数功能：计算每个缺失值所对应的k近邻的隶属度
def calculate_membership(fpr, miss_index, knn_index, k_index, T):
    """
    函数功能：计算每个缺失值所对应的k近邻的隶属度
    :param fpr: 模糊偏好关系（现在是不完备的）
    :param miss_index: 缺失值索引
    :param knn_index: 对应的社交信任网络的近邻关系
    :param k_index: 缺失值对应的k近邻
    :return: 每个缺失值所对应的k近邻的隶属度
    """
    num, k = k_index.shape
    member = np.zeros((num, k))
    for i in range(num):
        # 处理特殊情况：当没有k个近邻的情况
        DM_index, row, column = miss_index[i]
        sum1 = np.sum(k_index[i] == -1)
        sum2 = k - sum1
        knn_knn_index = knn_index[k_index[i, :sum2], 1:k+1]
        for j in range(sum2):
            DM_knnindex = k_index[i, j]
            M = np.concatenate([[fpr[DM_index]]]*k, axis=0)
            M[M == -5] = 5
            M1 = np.abs(M - fpr[knn_knn_index[j]])
            row_sum = np.sum(M1 <= 1, axis=2)[:, row]
            column_sum = np.sum(M1 <= 1, axis=1)[:, column]
            M1[M1 > 1] = 0
            row_s = np.sum(M1, axis=2)[:, row]
            column_s = np.sum(M1, axis=1)[:, column]
            sim = 1 - (row_s+column_s)/(row_sum+column_sum)

            M2 = fpr[DM_index].copy()
            M2[M2 == -5] = 5
            M3 = np.abs(M2 - fpr[DM_knnindex])
            row_sum1 = np.sum(M3 <= 1, axis=1)[row]
            column_sum1 = np.sum(M3 <= 1, axis=0)[column]
            M3[M3 > 1] = 0
            row_s1 = np.sum(M3, axis=1)[row]
            column_s1 = np.sum(M3, axis=0)[column]
            sim1 = 1 - (row_s1+column_s1)/(row_sum1+column_sum1)

            # 计算隶属度
            sim2 = (sim1 + np.sum(sim))/(sum2 + 1)
            member[i, j] = (T[DM_index, DM_knnindex]+sim2)/(1+sim2)

    return member


# 主要函数一：给定不完备模糊偏好关系，信任网络，参数，输出完备的模糊偏好关系
def estimate(ifpr, T, k):
    """
    基于knn的缺失值估计方法
    :param ifpr: 不完备的模糊偏好关系
    :param T: 社交信任网络
    :param k: k近邻
    :return: 完备的模糊偏好关系fpr
    """
    p = T.shape[0]
    n = ifpr[0].shape[0]
    knn_index = knn(T)
    # 给缺失值位置设定成-5，方便后面区分
    fpr = -5 * np.ones((p, n, n))
    fpr[ifpr >= 0] = ifpr[ifpr >= 0]
    while np.sum(fpr < 0) > 0:
        # 得到缺失值的个数
        num_miss = int(np.sum(fpr < 0) / 2)
        #print("缺失值个数: ", num_miss)
        # 得到缺失值的索引
        miss_index = np.ones((num_miss, 3), dtype=int)
        index1 = np.array(np.where(fpr < 0)).T
        ii = 0
        for i in range(num_miss):
            while index1[ii][1] > index1[ii][2]:
                ii = ii + 1
            miss_index[i] = index1[ii]
            ii = ii + 1

        # 得到每个缺失值在社交信任网络下的k近邻索引（缺失位置已知）
        k_index = -1 * np.ones((num_miss, k), dtype=int)
        for i in range(num_miss):
            judge_k = 0
            for j in range(1, p):
                judge_index = np.array([knn_index[miss_index[i, 0], j], miss_index[i, 1], miss_index[i, 2]])
                if np.any(np.all(miss_index == judge_index, axis=1)):
                    continue
                else:
                    k_index[i, judge_k] = knn_index[miss_index[i, 0], j]
                    judge_k += 1
                if judge_k == k:
                    break
        #print("每个缺失值的索引 ", miss_index)
        #print("每个缺失值对应的kNN索引", k_index)
        # 得到每个缺失值对应的隶属度
        membership = calculate_membership(fpr, miss_index, knn_index, k_index, T)
        #print("每个缺失值的对k近邻的隶属度关系", membership)
        # 计算可靠度
        D = np.sum(membership, axis=1)
        #print("每个缺失值对应的可靠度", D)
        # 先修改可靠度最高的缺失值
        first_index = np.argmax(D)
        #print("当次修改的缺失值的索引", first_index)
        #print("当次估计缺失值的最高可靠度", D[first_index])
        membership1 = membership[first_index] / np.sum(membership[first_index])
        DM_index, row, column = miss_index[first_index]
        # fpr[DM_index, row, column] = np.round(membership1 @ fpr[k_index[first_index], row, column], decimals=2)
        fpr[DM_index, row, column] = membership1 @ fpr[k_index[first_index], row, column]
        fpr[DM_index, column, row] = 1 - fpr[DM_index, row, column]
        #print("估计缺失值的位置：", [DM_index, row, column])
        #print("估计缺失值的k近邻索引", k_index[first_index])
        #print("估计缺失值对k近邻的隶属度", membership1)
        #print("估计的模糊偏好关系的值", fpr[DM_index, row, column])
    return fpr


# 函数功能：基于信任网络计算每个决策者的信任影响
def calculate_In(T):
    p = T.shape[0]
    IF = np.sum(T, axis=0) / p

    return IF


# 函数功能：根据上面的信任影响计算初始聚类中心
def initial_centroid(T, sigma):

    p = T.shape[0]
    centroids = []
    theta = np.std(T)
    # 计算信任影响力
    IF = calculate_In(T)
    E1 = np.argsort(IF)[::-1]
    DM_index = 0
    centroids.append(E1[DM_index])
    DM_index = DM_index + 1
    while DM_index != p:
        d = 1 - (T[E1[DM_index], centroids] + T[centroids, E1[DM_index]])/2
        d_minindex = np.argmin(d)
        if d[d_minindex] > sigma * theta:
            centroids.append(E1[DM_index])
        DM_index = DM_index + 1

    return centroids


# 函数功能计算决策者集合和决策者集合的相互影响力
def calulate_MI(FPR, T, DMset1, DMset2):
    F1 = np.mean(FPR[DMset1], axis=0)
    F2 = np.mean(FPR[DMset2], axis=0)
    T1 = np.sum(np.sum(T[DMset1][:, DMset2])+np.sum(T[DMset2][:, DMset1]))/(2*len(DMset1)*len(DMset2))
    MI = T1/(1+np.square(np.linalg.norm(F1-F2, ord='fro')))

    return MI



# 修改函数：聚类的最终的终止条件
def im_kmeans(FPR, T, centroids, AC, alpha):

    p = T.shape[0]
    c_num = len(centroids)
    SSE_change = 1
    SSE0 = 0
    iter = 0
    centroids_new = centroids.copy()
    while SSE_change > 10e-2:
        cluster_result = []
        for i in range(c_num):
            cluster_result.append([centroids_new[i]])
        for i in range(p):
            if np.sum(np.isin(np.concatenate(cluster_result), i)) > 0:
                continue
            else:
                dis = np.zeros(c_num)
                for j in range(c_num):
                    DMset1 = [i]
                    DMset2 = [centroids_new[j]]
                    MI = calulate_MI(FPR, T, DMset1, DMset2)
                    AC1 = AC[i]
                    AC2 = AC[centroids_new[j]]
                    dis[j] = alpha * (1 - MI) + (1 - alpha) * np.abs(AC2 - AC1)
                index = np.argmin(dis)
                cluster_result[index].append(i)

        # 每个决策者都被分好，计算聚类效果SSE
        dis1 = 0  # 记录子组间的距离
        for i in range(c_num):
            for j in range(1, len(cluster_result[i])):
                DMset1 = [cluster_result[i][j]]
                DMset2 = [centroids_new[i]]
                MI = calulate_MI(FPR, T, DMset1, DMset2)
                AC1 = AC[cluster_result[i][j]]
                AC2 = AC[centroids_new[i]]
                dis1 = dis1 + (alpha * (1 - MI) + (1 - alpha) * np.abs(AC2 - AC1))**2

        SSE = dis1
        SSE_change = np.abs(SSE - SSE0)
        SSE0 = SSE
        iter = iter + 1
        if iter > 100:
            break

        if SSE_change > 10e-2:
            # 更新类中心
            centroids_new = []
            for i in range(c_num):
                dis = np.zeros(len(cluster_result[i]))
                for j in range(len(cluster_result[i])):
                    DMset1 = [cluster_result[i][j]]
                    MI = calulate_MI(FPR, T, DMset1, cluster_result[i])
                    AC1 = AC[cluster_result[i][j]]
                    AC2 = np.sum(AC[cluster_result[i]]) / (len(cluster_result[i]))
                    dis[j] = alpha * (1 - MI) + (1 - alpha) * np.abs(AC1 - AC2)
                index = np.argmin(dis)
                centroids_new.append(cluster_result[i][index])

    return cluster_result


# 函数功能：给定模糊偏好关系矩阵和权重计算共识
def calculate_CD(fpr, psi):

    p = len(psi)
    n = fpr[0].shape[0]
    Psi = np.repeat(np.repeat(psi, n, axis=0).reshape(p, n), n, axis=0).reshape(p, n, n)
    center = np.sum(Psi * fpr, axis=0)
    sub_CD = np.zeros(p)
    for i in range(p):
        sub_CD[i] = 1 - (np.sum(np.abs(center - fpr[i]))) / (n * (n-1))
    CD = sub_CD @ psi
    return CD


# 函数功能：计算不完备模糊偏好关系矩阵的共识
def calculate_in_CD(ifpr, psi):

    p = len(psi)
    n = ifpr[0].shape[0]
    Psi = np.repeat(np.repeat(psi, n, axis=0).reshape(p, n), n, axis=0).reshape(p, n, n)
    fpr = -5 * np.ones((p, n, n))
    fpr[ifpr >= 0] = ifpr[ifpr >= 0]
    Psi[fpr < 0] = 0
    Psi1 = Psi.copy()
    for i in range(p):
        Psi1[i] = Psi[i] / np.sum(Psi, axis=0)
    fpr[fpr < 0] = 0
    center = np.sum(Psi1 * fpr, axis=0)
    CD = 0
    for i in range(p):
        R = Psi1[i] * np.abs(fpr[i] - center)
        CD = CD + np.sum(R[ifpr[i] >= 0]) / (np.sum(ifpr[i] >= 0) - n)

    CD = 1-CD
    return CD


# 函数功能：给定模糊偏好关系矩阵，聚类中心，理想阈值，输出正域
# 这个函数返回的是在给定决策者集合中在正域决策者的索引。
def calculate_pos(FPR, center, theta_ideal):

    p = len(FPR)
    n = FPR[0].shape[0]
    p_set = np.arange(0, p, 1)
    judge = 1 - np.sum(np.sum(np.abs(FPR-center), axis=2), axis=1)/(n*(n-1))
    # std = np.std(FPR, axis=0) + epsilon
    # judge = np.sum(np.sum(np.abs(FPR-center) > std, axis=1), axis=1)
    Pos = p_set[judge >= theta_ideal]
    if len(Pos) == 0:
        Pos = np.array([-1])

    return Pos


# 函数功能：计算不可边度
def calculate_1_CAP(DM_remain, Pos_T, AC, FPR, center, beta, gamma, DM_remain_index):
    """
    函数功能，计算修改优先级，不过计算的是 1-CAP
    :param DM_remain: 当前子组除了正域之外的所有决策者
    :param Pos_T: 剩余决策者对正域的信任
    :param AC: 单位调整成本
    :param FPR: 完备模糊偏好关系
    :param center: 群体意见
    :param beta: 控制调整成本的参数
    :param gamma: 现在已经没有这个参数了
    :param DM_remain_index: 剩余决策者在原始模糊偏好关系中的索引
    :return: 剩余决策者的每个 1-CAP，方便后续划分边界域和负域
    """
    T_part = gamma * (1-np.mean(Pos_T, axis=1))
    AC_part = beta * AC[DM_remain].reshape(-1)
    n = len(center[0])
    df = 1 - np.sum(np.sum(np.abs(FPR[DM_remain_index]-center), axis=1), axis=1)/(n*(n-1))
    df_part = (1-beta-gamma) * df / np.max(df)
    CAP1 = T_part + AC_part + df_part

    return CAP1


# 主要函数三：共识达成过程
def consensus_reach_dynamic(FPR, DM, T, AC, beta, gamma, psi, theta_ideal, lammda, epsilon, xi):
    """
    函数旨在达成共识
    :param FPR:一组模糊偏好关系；在第一阶段为组内，第二阶段为各个组的类中心模糊偏好关系
    :param T: 社区信任网络
    :param AC: 单位调整成本
    :param beta: 单位调整成本的控制参数
    :param gamma:信任度的控制参数
    :param psi:各个决策者的权重，在第一阶段每个决策者的权重相等，第二阶段根据子组大小确定
    :return:达成共识的结果
    """
    # 先根据权重计算类中心
    p = len(psi)
    n = FPR[0].shape[0]
    Psi = np.repeat(np.repeat(psi, n, axis=0).reshape(p, n), n, axis=0).reshape(p, n, n)
    center = np.sum(Psi * FPR, axis=0)
    #print("群体意见", center)
    T1 = T.copy()
    # 根据权重计算共识度
    CD = calculate_CD(FPR, psi)
    #print("当前的共识", CD)
    center_new = center.copy()
    FPR_new = FPR.copy()
    # 当共识未达到阈值时
    iter = 0
    while CD < theta_ideal:

        # 计算正域及相关信任度
        Pos = calculate_pos(FPR_new, center_new, theta_ideal)
        DM1 = np.array(DM)
        #print("属于正域的决策者", DM1[Pos])
        if Pos[0] == -1:
            DM_remain = DM1
            Pos_T = np.zeros((len(DM_remain), len(Pos)))
            Pos_T[:, 0] = np.mean(T1[DM][:, DM])
            Pos_FPR = center_new.reshape(1, n, n)
        else:
            DM_remain = np.array(list(set(DM) - set(DM1[Pos])))
            Pos_T = T1[DM_remain][:, Pos]
            if len(Pos) == 1:
                Pos_FPR = FPR_new[Pos].reshape(1, n, n)
            else:
                Pos_FPR = FPR_new[Pos]
        #print("剩余的决策者", DM_remain)
        # 计算不可变度
        DM_remain_index = np.array(list(set(np.arange(0, len(DM), 1)) - set(Pos)))
        CAP1 = calculate_1_CAP(DM_remain, Pos_T, AC, FPR_new, center_new, beta, gamma, DM_remain_index)
        CAP1_lammda = np.min(CAP1) + lammda * (np.max(CAP1) - np.min(CAP1))
        # 计算边界域和负域的决策者
        Neg = DM_remain[CAP1 <= CAP1_lammda]
        #print(Bon)

        # 约束条件
        Neg_index = []
        Neg_index1 = []
        for i in range(len(Neg)):
            Neg_index.append(list(np.where(DM == Neg[i])[0]))
            Neg_index1.append(list(np.where(DM_remain == Neg[i])[0]))
        Neg_index = np.array(Neg_index).reshape(-1)
        Neg_index1 = np.array(Neg_index1).reshape(-1)
        p1 = len(Neg)
        #print(Bon_index)
        Pos_T1 = Pos_T[Neg_index1] / np.sum(Pos_T[Neg_index1], axis=1).reshape(p1, 1)
        F1 = FPR_new[Neg_index]
        p2 = len(Pos)
        epsilon_matrix = epsilon * np.ones((p1, n, n))
        for i in range(p1):
            matrix = np.repeat(np.repeat(Pos_T1[i], n, axis=0).reshape(p2, n), n, axis=0).reshape(p2, n, n)
            F1[i] = np.sum(matrix * Pos_FPR, axis=0)
            np.fill_diagonal(epsilon_matrix[i], 0)
        F1_flat = F1.reshape(-1)
        #print(F1_flat)
        epsilon_matrix_flat = epsilon_matrix.reshape(-1)
        #print(epsilon_matrix_flat)
        x_len = len(F1_flat)
        C1 = np.zeros(x_len)
        AC2 = np.repeat(AC[Neg], n*n)
        C2 = np.ones(x_len) * AC2
        C = np.concatenate((C1, C2), axis=0)# 目标函数的系数
        #print(C)
        # 下面写约束条件的矩阵形式
        A1 = -1 * np.eye(x_len)
        A2 = np.eye(x_len)
        A3 = np.concatenate((A2, A1), axis=1)
        A4 = np.concatenate((A1, A1), axis=1)
        A5 = np.zeros((x_len, x_len))
        A6 = np.concatenate((A2, A5), axis=1)
        A7 = np.concatenate((A1, A5), axis=1)
        A = np.concatenate((A3, A4, A6, A7), axis=0)
        #print(A)
        # 不等式约束右边的常熟向量
        b1 = FPR_new[Neg_index].reshape(-1)
        b2 = np.concatenate((b1, -b1), axis=0)
        b3 = F1_flat + epsilon_matrix_flat
        b4 = epsilon_matrix_flat - F1_flat
        b = np.concatenate((b2, b3, b4), axis=0)
        #print(b)
        result = linprog(C, A_ub=A, b_ub=b, method='highs')
        FPR_new[Neg_index] = result.x[:x_len].reshape(p1, n, n)

        center_new = np.sum(Psi * FPR_new, axis=0)

        # 根据权重计算共识度
        CD = calculate_CD(FPR_new, psi)
        #print("当前共识为", CD)
        #print(FPR_new)
        iter = iter + 1
        if iter > 10:
            break
        # 动态更新信任度
        if Pos[0] == -1:
            continue
        for i in range(p1):
            for j in range(p2):
                #print("负域：", DM1[Bon_index[i]])
                #print("正域中决策者", DM1[Pos[j]])
                #print("原先的信任", T1[DM1[Bon_index[i]], DM1[Pos[j]]])
                T1[DM1[Neg_index[i]], DM1[Pos[j]]] = np.power(T1[DM1[Neg_index[i]], DM1[Pos[j]]], (1-T1[DM1[Neg_index[i]], DM1[Pos[j]]]))
                #print("修改后的信任", T1[DM1[Bon_index[i]], DM1[Pos[j]]])
        # 此外，还得更新负域中决策者对边界域中决策者的信任度
        Bon_index = np.array(list(set(np.arange(0, len(DM), 1)) - set(Pos) - set(Neg_index)))
        for i in range(len(Bon_index)):
            for j in range(p1):
                #print("边界域：", DM1[Neg_index[i]])
                #print("负域中决策者", DM1[Bon_index[j]])
                #print("原先的信任", T1[DM1[Neg_index[i]], DM1[Bon_index[j]]])
                T1[DM1[Bon_index[i]], DM1[Neg_index[j]]] = np.power(T1[DM1[Bon_index[i]], DM1[Neg_index[j]]], 1-np.mean(T1[DM1[Bon_index[i]], DM1[Neg_index[j]]]))
                #print("修改后的信任", T1[DM1[Neg_index[i]], DM1[Bon_index[j]]])
    print("iter:",iter)
    T = T1
    #print(center_new)
    return FPR_new, center_new, CD, T


# 函数功能：基于最后的聚类中心，决定最终选择什么方案
def select_alternative(center):

    score = np.sum(center, axis=1)
    #print("每个方案的得分", score)
    order = np.argsort(score)

    return order[::-1]


# 总函数：整个方法的流程
def all_progress(IFPR, T, knn_k, AC, alpha, beta, gamma, theta_ideal, lammda, epsilon, xi, sigma):

    FPR = estimate(IFPR, T, knn_k)
    p = len(FPR)
    n = FPR[0].shape[0]
    FPR_end = np.zeros((p, n, n))
    initial_centroids = initial_centroid(T, sigma)
    cluster_result = im_kmeans(FPR, T, initial_centroids, AC, alpha)
    #print(cluster_result)
    FPR1 = []
    for i in range(len(cluster_result)):
        FPR1.append([])
        FPR1[i].append(FPR[cluster_result[i]])
    FPR1_new = FPR1.copy()
    center1 = np.ones((len(cluster_result), n, n))
    T_1 = T.copy()
    # 第一阶段的共识达成
    for i in range(len(cluster_result)):
        psi = np.ones(len(cluster_result[i])) / len(cluster_result[i])
        p1 = len(cluster_result[i])
        FPR1_new[i][0], _, CD_, T_1 = consensus_reach_dynamic(FPR1[i][0], cluster_result[i], T_1, AC, beta, gamma, psi, theta_ideal,
                                                   lammda,
                                                   epsilon, xi)
        # print(CD_)
        FPR_end[cluster_result[i]] = FPR1_new[i][0]
        Psi = np.repeat(np.repeat(psi, n, axis=0).reshape(p1, n), n, axis=0).reshape(p1, n, n)
        center1[i] = np.sum(Psi * FPR1_new[i][0], axis=0)
    # 第二阶段的共识达成
    #print("第二阶段开始")
    psi = np.ones(len(cluster_result))
    AC1 = np.ones(len(cluster_result))
    T1 = np.ones((len(cluster_result), len(cluster_result)))
    for i in range(len(cluster_result)):
        psi[i] = len(cluster_result[i]) / p
        AC1[i] = np.mean(AC[cluster_result[i]])
        for j in range(len(cluster_result)):
            if i == j:
                T1[i, j] = 1
            else:
                T1[i, j] = np.mean(T_1[cluster_result[i]][:, cluster_result[j]])
    #print("先输出信任度", T1)
    new_cluster = np.arange(0, len(cluster_result), 1)
    center1_new, Center, CD, T1 = consensus_reach_dynamic(center1, new_cluster, T1, AC1, beta, gamma, psi, theta_ideal, lammda,
                                            epsilon, xi)

    #print(CD)
    return center1_new, Center, CD, FPR, FPR_end, center1, T1, T_1



if __name__ == '__main__':
    # 自己方法的一些参数设定
    sigma = 1  # 选择初始聚类中心时候标准差前面的系数
    alpha = 0.5  # 计算相互影响力时候，控制调整成本的影响的参数
    beta = 0.4  # 计算不可变度时，控制调整成本的参数
    gamma = 0.5  # 计算不可变度时，控制信任度的参数
    lammda = 0.5  # 边界域和负域的划分比
    epsilon = 0.02  # 允许修改的最大量
    theta_ideal = 0.92  # 理想的共识度
    xi = 0
    knn_k = 4  # 通过main1实验分析得到的knn参数


    IFPR_pd = pd.read_excel('incomplete_data.xlsx', header=None)
    T_pd = pd.read_excel('trust.xlsx', header=None)
    p = 20
    n = 5
    IFPR = np.array(IFPR_pd).reshape(p, n, n)
    T = np.array(T_pd)
    IFPR[IFPR == '~'] = np.nan
    IFPR = np.array(IFPR, dtype=float)
    np.random.seed(42)
    AC = np.random.rand(p)
    AC = np.round(AC, 2)

    #FPR = estimate(IFPR, T, knn_k)
    #print(FPR)
    FPR_new, center_new, CD, _, _, _, T1, T_1 = all_progress(IFPR, T, knn_k, AC, alpha, beta, (1-beta)/2, theta_ideal, lammda, epsilon, xi, sigma)
    order = select_alternative(center_new)
    print(order)