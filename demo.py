import pandas as pd
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import pairwise_distances
from pprint import pprint
import numpy as np

""" pip install pandas
    pip install scikit_learn
    pip config set global.index-url https://pypi.douban.com/simple/ 设置源，加速下载
"""

users = ["User1", "User2", "User3", "User4", "User5"]
items = ["Item A", "Item B", "Item C", "Item D", "Item E"]
# 构建数据集 1 买  0 不买
datasets = [
    [1, 0, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
]
"""创建dataframe表格数据结构"""
df = pd.DataFrame(datasets, columns=items, index=users)
print(df)

print(df.dtypes)

# 计算Item A 和Item B的相似度
print(jaccard_score(df["Item A"], df["Item B"]))

"""
Jaccard相似度： https://www.lianxh.cn/news/47fc90b1c540e.html
 基于User-Based CF
"""
# 计算杰卡尔德相似度=1-杰卡尔德距离
user_similar = 1 - pairwise_distances(df.values, metric='jaccard')
user_similar = pd.DataFrame(user_similar, columns=users, index=users)
print("用户之间的两两相似度：")
print(user_similar)

topN_users = {}
# 遍历每一行数据
for i in user_similar.index:
    # 取出每一列数据，并删除自身，然后排序数据 drop i 把自己去掉
    _df = user_similar.loc[i].drop([i])
    # 进行降序排序
    _df_sorted = _df.sort_values(ascending=False)
    # 抓出前两个排名的
    top2 = list(_df_sorted.index[:2])
    topN_users[i] = top2

print("Top2相似用户：")
pprint(topN_users)

rs_results = {}
# 构建推荐结果，遍历字典 key,value
"""
{'User1': ['User3', 'User2'],
 'User2': ['User1', 'User4'],
 'User3': ['User1', 'User5'],
 'User4': ['User2', 'User5'],
 'User5': ['User3', 'User1']}

 df：
        Item A  Item B  Item C  Item D  Item E
User1       1       0       1       1       0
User2       1       0       0       1       1
User3       1       0       1       0       0
User4       0       1       0       1       1
User5       1       1       1       0       1


"""
for user, sim_users in topN_users.items():
    rs_result = set()  # 存储推荐结果
    for sim_user in sim_users:
        # 构建初始的推荐结果
        rs_result = rs_result.union(set(df.loc[sim_user].replace(0, np.nan).dropna().index))
    # 过滤掉已经购买过的物品
    rs_result -= set(df.loc[user].replace(0, np.nan).dropna().index)
    rs_results[user] = rs_result
print("最终推荐结果：")
pprint(rs_results)

"""
Item-Based CF
"""
# 计算物品间相似度
item_similar = 1 - pairwise_distances(df.T.values, metric="jaccard")
item_similar = pd.DataFrame(item_similar, columns=items, index=items)
print("物品之间的两两相似度：")
print(item_similar)

topN_items = {}
# 遍历每一行数据
for i in item_similar.index:
    # 取出每一列数据，并删除自身，然后排序数据
    _df = item_similar.loc[i].drop([i])
    _df_sorted = _df.sort_values(ascending=False)

    top2 = list(_df_sorted.index[:2])
    topN_items[i] = top2

print("Top2相似物品：")
pprint(topN_items)

rs_results = {}
# 构建推荐结果
for user in df.index:  # 遍历所有用户
    rs_result = set()
    for item in df.loc[user].replace(0, np.nan).dropna().index:  # 取出每个用户当前已购物品列表
        # 根据每个物品找出最相似的TOP-N物品，构建初始推荐结果
        rs_result = rs_result.union(topN_items[item])
    # 过滤掉用户已购的物品
    rs_result -= set(df.loc[user].replace(0, np.nan).dropna().index)
    # 添加到结果中
    rs_results[user] = rs_result

print("最终推荐结果：")
pprint(rs_results)
