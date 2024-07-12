import numpy as np
from matplotlib import pyplot as plt

class func_Utility:
    def u_t(gamma, Rt, Rmin, Rmax):
        '''
        技术状况效用
        '''
        if Rmin == Rmax:
            print('最大值与最小值不能相等')
            pass
        else:
            out = 1/(1-np.exp(-gamma)) * (1-np.exp(-gamma*((Rt-Rmin)/(Rmax-Rmin))))
            return out
    
    def u_i(gamma, Ri, Rmin, Rmax):
        '''
        可持续效用分量
        '''
        if Rmin == Rmax:
            print('最大值与最小值不能相等')
            pass
        else:
            out = 1/(1-np.exp(-gamma)) * (1-np.exp(-gamma*((Rmax-Ri)/(Rmax-Rmin))))
            return out

    def u_c(gamma, Cmaint, Cmax):
        '''
        成本效用
        '''
        out = 1/(1-np.exp(-gamma)) * (1-np.exp(-gamma*((Cmax-Cmaint)/Cmax)))
        return out

class scheme():
    '''
    tech: 技术评分增加值
    cost: 支出费用（万元）
    evan: 可持续成本（万元）
    '''
    def __init__(self):
        self.tech = None
        self.cost = None
        self.evan = None

    def score_cal(init, scheme_idx, tech_up, tech_down, tech_deck):
        '''
        init: 桥梁初始技术评分（部件级），二维index
        scheme_idx: 方案的索引
        tech_up: 各方案所增加的技术评分,即scheme_up.tech
        '''
        score_sum = []
        for i in range(len(scheme_idx)):
            score1 = 0.4 * (init[0] + tech_up[scheme_idx[i][0]])
            score2 = 0.4 * (init[1] + tech_down[scheme_idx[i][1]])
            score3 = 0.2 * (init[2] + tech_deck[scheme_idx[i][2]])
            score_sum.append(score1 + score2 + score3)
        return score_sum

class Search:
    def __init__(self):
        self.tech_origin = None
        self.sum_tech = []
        self.sum_cost = []
        self.sum_evan = []
        self.score_st = 80
        self.cost_st = 56
        self.idx_left = []
        self.index = []
        self.gamma = 1
        self.ei = np.array([[1/3, 1/3, 1/3]])
        self.gamma = 1
        self.scheme_up = []
        self.scheme_down = []
        self.scheme_deck = []


    def execute(self):
        s = 0
        for i in range(len(self.scheme_up.tech)):
            for j in range(len(self.scheme_down.tech)):
                for k in range(len(self.scheme_deck.tech)):
                    self.sum_tech.append((self.tech_origin[0] + self.scheme_up.tech[i])*0.4 +
                                    (self.tech_origin[1] + self.scheme_down.tech[j])*0.4 + 
                                    (self.tech_origin[2] + self.scheme_deck.tech[k])*0.2)
                    self.sum_cost.append(self.scheme_up.cost[i] + self.scheme_down.cost[j] + self.scheme_deck.cost[k])
                    self.sum_evan.append(self.scheme_up.evan[i] + self.scheme_down.evan[j] + self.scheme_deck.evan[k])
                    s = s+1
                    self.index.append([i,j,k])

        Score_all = scheme.score_cal(self.tech_origin, self.index, self.scheme_up.tech, self.scheme_down.tech, self.scheme_deck.tech)

        for i in range(len(Score_all)):
            if Score_all[i] > self.score_st and self.sum_cost[i] < self.cost_st:
                self.idx_left.append(i)
        return self

    def U_calculate(self):
        self.execute()
        ei = np.array([0.357615, 0.317178, 0.325207]).reshape(-1,1)
        uti_tech = func_Utility.u_t(gamma=self.gamma, 
                            Rt=np.array(self.sum_tech), 
                            Rmin=min(self.sum_tech), 
                            Rmax=max(self.sum_tech)).reshape(-1,1)
        uti_cost = func_Utility.u_c(gamma=self.gamma, 
                            Cmaint = np.array(self.sum_cost), 
                            Cmax = self.cost_st).reshape(-1,1)
        uti_evan = func_Utility.u_i(gamma=self.gamma, 
                            Ri=np.array(self.sum_evan), 
                            Rmin=min(self.sum_evan), 
                            Rmax=max(self.sum_evan)).reshape(-1,1)
        uti = np.concatenate([uti_tech, uti_cost, uti_evan], axis=1)

        U = np.dot(uti, ei)
        U = U[self.idx_left] # 仅保留技术评分大于80的U值
        uti_cost = uti_cost[self.idx_left]
        U_uti_cost = np.concatenate([U, uti_cost], axis=1)
        index = np.array(self.index)
        idx_left = np.array(self.idx_left)
        return U_uti_cost, uti, index, idx_left


# # 实例化Search
# search = Search()

# # 定义初始技术评分tech_origin、综合权重ei和风险偏好gamma
# search.tech_origin = [70, 80, 75]
# search.ei =  np.array([[0.357615, 0.317178, 0.325207]])
# search.gamma = 1

# # 定义养护方案
# scheme_up = scheme()
# scheme_up.tech = [0, 5, 10, 5, 10, 5]
# scheme_up.cost = [0, 20, 15, 10, 18, 16]
# scheme_up.evan = [0, 100, 150, 200, 120, 160]

# scheme_down = scheme()
# scheme_down.tech = [0, 10, 5, 10, 5, 5]
# scheme_down.cost = [0, 16, 18, 17, 15, 15]
# scheme_down.evan = [0, 123, 210, 150, 120, 100]

# scheme_deck = scheme()
# scheme_deck.tech = [0, 10, 15, 10]
# scheme_deck.cost = [0, 18, 20, 20]
# scheme_deck.evan = [0, 144, 135, 130]

# # 计算综合效用和成本效用
# U_uti_cost, uti, index, idx_left = search.U_calculate()
# #%%
# print(type(U_uti_cost))
# #%%
# # 可视化
# plt.figure()
# plt.scatter(U_uti_cost[:,0], U_uti_cost[:,1])
# # np.savetxt('uti_cost.txt', U_uti_cost[:,1], fmt='%.8f')
# # np.savetxt('U.txt', U, fmt='%.8f')

# eta = 0.5
# target = eta*U_uti_cost[:,0] + (1-eta)*U_uti_cost[:,1]

# plt.figure()
# plt.scatter(range(len(target)),target)
# np.savetxt('target.txt',target, fmt='%.8f')

# # %%
# idx_sort = np.argsort(target, axis=0) # 从小到达排序
# target_sort = target[idx_sort]
# #%%
# # 输出最优方案的索引值
# # 解释：idx_sort[-1,0]是最优方案在idx_left中的位置索引。idx_left存储的是筛选后的方案在所有方案中的索引。index储存了所有方案的[i,j,k]位置
# optimal_scheme = index[idx_left[idx_sort[-1]]]

# Cost = scheme_up.cost[optimal_scheme[0]] + scheme_down.cost[optimal_scheme[1]] + scheme_deck.cost[optimal_scheme[2]]
# Tech_score = scheme.score_cal(search.tech_origin, [optimal_scheme], scheme_up.tech, scheme_down.tech, scheme_deck.tech)
# print('上部结构：选第{0}种养护方案\n下部结构：选第{1}种养护方案\n桥面系:选第{2}种养护方案'.format(optimal_scheme[0]+1,optimal_scheme[1]+1,optimal_scheme[2]+1))
# print('\n')
# print('最优方案的总维养花费为{0}万元'.format(Cost))
# print('最优方案的最终技术评分为{0}'.format(Tech_score[0]))



# %%
