import pandas as pd
import pulp
import numpy as np

# 加载数据
df_tjk = pd.read_csv("df_matrix.csv")
df_q = pd.read_csv("df_order.csv")
df_p = pd.read_csv("df_proc.csv")

# 提取必要的数组
q = df_q.values[:, 1].reshape(-1, 1)  # 提取 df_order.csv 的第二列作为 q
p = df_p.values[:, 2].reshape(-1, 1)  # 提取 df_proc.csv 的第三列作为 p
s = df_p.values[:, 1].reshape(-1, 1)  # 提取 df_proc.csv 的第二列作为 s
T = df_tjk.values[:, 1:]
T = np.hstack((T, T))
  # 从 df_matrix.csv 中提取第2列之后的所有列作为 T 矩阵

# 定义线性规划问题
problem = pulp.LpProblem("Linear_Programming_Problem", pulp.LpMinimize)

# 定义变量
M = pulp.LpVariable.dicts("M", ((j, k) for j in range(1, 21) for k in range(1, 213)), cat="Binary")
z = pulp.LpVariable.dicts("z", range(1, 21), cat="Binary")
z_plus = pulp.LpVariable.dicts("z_plus", range(1, 21), cat="Binary")
y = pulp.LpVariable.dicts("y", range(1, 21), lowBound=0)  # 辅助变量 y_j
u = pulp.LpVariable.dicts("u", range(1, 21), lowBound=0)  # 辅助变量 u_j
v = pulp.LpVariable.dicts("v", range(1, 21), lowBound=0)  # 辅助变量 v_j 表示乘积结果

# 定义二元变量 I_k
I = pulp.LpVariable.dicts("I", range(1, 213), cat="Binary")

# 目标函数
objective = (
    1.85 * pulp.lpSum((1 - M[19, k] - M[20, k]) * q[k-1][0] for k in range(1, 213)) +
    1.25 * pulp.lpSum((M[19, k] + M[20, k]) * q[k-1][0] for k in range(1, 213)) +
    250000 * pulp.lpSum(z[j] for j in range(1, 21)) +
    pulp.lpSum(10000 * v[j] for j in range(1, 21)) +
    10000 * pulp.lpSum(z_plus[j] for j in range(1, 21))
)
problem += objective

# 约束条件

# 约束1: z[j] 直接与 M[j, k] 关联
for j in range(1, 21):
    for k in range(1, 213):
        problem += z[j] >= M[j, k]

problem += z[7] == 1
problem += z[12] == 1

# 约束2: 固定 Mjk 的值
for k in range(1, 107):
    problem += M[12, k] == 0
for k in range(107, 213):
    problem += M[7, k] == 0

# 约束3: Mjk 每列有且仅有一个1
for k in range(1, 213):
    problem += pulp.lpSum(M[j, k] for j in range(1, 21)) == 1

# 约束4: z_plus 只能在 z[j] = 1 时等于1
for j in range(1, 21):
    problem += z_plus[j] <= z[j]

# 约束5: 容量约束
for j in range(1, 21):
    problem += pulp.lpSum(M[j, k] * q[k-1][0] for k in range(1, 213)) <= s[j-1][0]

# 约束6: 辅助变量 u[j] 的约束 (表示 u[j] = p[j] * sum(M[j,k] * q[k] for k))
for j in range(1, 21):
    problem += u[j] == p[j-1][0] * pulp.lpSum(M[j, k] * q[k-1][0] for k in range(1, 213))

# 约束7: 辅助变量 v[j] 的约束 (表示 v[j] = u[j] * (1 - z_plus[j]))
for j in range(1, 21):
    problem += v[j] <= u[j]
    problem += v[j] <= 10000 * (1 - z_plus[j])
    problem += v[j] >= u[j] - 10000 * z_plus[j]

# 约束8: 比例约束
#这里保证了如果
bigM = 60000  # 确保 bigM 的选择符合问题的规模

for k in range(1, 213):
    # 改进的逻辑：使用一个新的变量来处理条件
    constraint_value = 600 - pulp.lpSum(M[j, k] * T[j-1][k-1] for j in range(1, 21))
    # 条件1：如果 constraint_value > 0，则 I[k] = 1；否则 I[k] = 0
    problem += constraint_value <= bigM * I[k]
    # 条件2：确保 I[k] 的正确值
    problem += constraint_value >=0 - bigM * (1 - I[k])


problem += pulp.lpSum(I[k] * q[k-1][0] for k in range(1, 213)) / pulp.lpSum(q[k-1][0] for k in range(1, 213)) >= 0.95
# constraint_value>=0  Ik必须等于 1，这表示条件满足。
# constraint_value<0  Ik必须等于 0，这表示条件不满足。
# 求解问题
problem.solve()

# 输出结果并导出到 Excel
M_values = []

for j in range(1, 21):
    row = []
    for k in range(1, 213):
        value = M[j, k].varValue
        row.append(int(round(value)))  # 确保输出为整数
    M_values.append(row)

# 转换为 DataFrame
df_M = pd.DataFrame(M_values, index=[f"j{j}" for j in range(1, 21)], columns=[f"k{k}" for k in range(1, 213)])

# print("Optimal Solution to the problem:")
# for j in range(1, 21):
#     for k in range(1, 213):
#         print(f"M[{j},{k}] = {int(round(M[j, k].varValue))}")

print("z values:")
for j in range(1, 21):
    print(f"z[{j}] = {int(round(z[j].varValue))}")

print("z_plus values:")
for j in range(1, 21):
    print(f"z_plus[{j}] = {int(round(z_plus[j].varValue))}")

print("I values:")
for k in range(1, 213):
    # 假设 I 是一个已经定义并求解过的 pulp 二进制变量列表或字典
    # 使用 pulp.value() 函数来获取变量的解值，并四舍五入
    I_value = int(round(pulp.value(I[k])))
    # 根据 I_value 的值输出不同的文本说明
    if I_value == 0:
        print(f"I[{k}] = {I_value}, 第{k}个订单不满足")
    else:
        print(f"I[{k}] = {I_value}, 第{k}个订单满足")

# 保存到 Excel
df_M.to_excel("M_values.xlsx", index=True)

print("结果已导出到 M_values.xlsx")


# 计算分子和分母
numerator = pulp.lpSum((pulp.value(I[k])) * q[k-1][0] for k in range(1, 213))
denominator = pulp.lpSum(q[k-1][0] for k in range(1, 213))
result = numerator / denominator
print("计算的结果:", result)
