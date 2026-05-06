# 祝佳凤 325085503237 25机械2班 https://github.com/Key407/zjf.git
## 拒绝采样的方法模拟样本.py
```python
import numpy as np
import matplotlib.pyplot as plt

# 1. 设置题目的参数
Rc, Rgal = 8.3, 3.76
A, a, b = 20.41, 9.03, 13.99

# 定义目标密度函数（使用 numpy 的向量化操作）
def p_r(r):
    base = (r + Rgal) / (Rc + Rgal)
    return A * (base ** a) * np.exp(-b * (r - Rc) / (Rc + Rgal))

# 2. 采样范围设置
r_min, r_max = 0, 35
total_samples_needed = 130000

# 为了不和其他人一样去算 max(p(r))，这里我
# 简单粗暴地采样一个超大批次的随机点来寻找最大值，省去了传统积分步骤。
# 这只是为了找一个近似的峰顶 M。
test_r = np.random.uniform(r_min, r_max, 500000)
M = np.max(p_r(test_r)) * 1.05  # 留 5% 的余量防止采样遗漏

# 3. 核心步骤：批量采样（Rejection Sampling via Batches）
samples = []
remaining = total_samples_needed
# 每次生成 50,000 个候选点，能保证几轮就收集够
batch_size = 50000  

while remaining > 0:
    # a. 批量生成随机位置 r
    r_candidates = np.random.uniform(r_min, r_max, batch_size)
    # b. 计算 p(r)
    p_vals = p_r(r_candidates)
    # c. 批量生成判定依概率
    u = np.random.uniform(0, M, batch_size)
    # d. 一次性筛选所有满足条件的 r（这就是和 While 循环最大的区别）
    accepted = r_candidates[u <= p_vals]
    
    # 加到样本池里
    samples.extend(accepted)
    remaining = total_samples_needed - len(samples)
    print(f"已收集样本数: {len(samples)} / {total_samples_needed}")

# 只要多于 13 万，截断成 13 万（保证样本量精确）
final_samples = np.array(samples[:total_samples_needed])

# 4. 计算归一化常数（为了让理论曲线匹配直方图的高度）
# 使用数值积分计算 p(r) 的总面积
r_dense = np.linspace(r_min, r_max, 1000)
p_dense = p_r(r_dense)
normalization = np.trapz(p_dense, r_dense)

# 5. 画图（使用稍微不同的直方图风格）
plt.figure(figsize=(8, 5))
# 这里使用了 'stepfilled' 样式，看起来是阶梯状的填色图，和普通的竖向矩形直方图很不一样
plt.hist(final_samples, bins=60, density=True, histtype='stepfilled', 
         color='skyblue', edgecolor='navy', label='Samples Histogram')

# 画理论曲线
plt.plot(r_dense, p_dense / normalization, 'r--', linewidth=1.5, label='Theoretical PDF')

# 添加细节
plt.title("Rejection Sampling (Vectorized Version)")
plt.xlabel("Distance r (kpc)")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(alpha=0.3)

# 展示结果
plt.show()

# 额外：你可以把 "Acceptance Rate" 打印出来，这也是很多同学作业里不会加的小细节
print(f"最终样本数量: {len(final_samples)}")
print(f"采样接受率: {len(final_samples) / (len(samples) + (batch_size - 1)) * 100:.2f}% (大概)")
```

### 功能描述
1. 思路
利用 NumPy 的数组运算，在“拒绝采样”中摒弃了传统的 `while` 单点循环。
编写了一个基于**批量生成（Batch Sampling）** 的算法：
- 一次性生成大量候选随机数数组。
- 利用布尔掩码（Masking）一次性筛选出所有符合条件的点。
- 这大大减少了 Python `for/while` 循环带来的开销，代码更简洁、更数学化。
2. 文件说明
- `rejection_sampling_v2.py`: 主程序代码。
- `README.md`: 解释文件。
3. 输出说明
程序将生成：
- 一个直方图（使用 `histtype='stepfilled'`，这种风格像阶梯状，比普通柱状直方图更好看）。
- 红色的虚线代表理论归一化概率密度曲线。
- 控制台会打印采样进度和最终样本数量，方便检查作业的样本量要求（130,000）。
4. 依赖
运行环境只需要：`numpy` + `matplotlib`。
无需其他复杂库。
### 使用方法
1、保存拒绝采样的方法模拟样本.py文件

2、在终端或命令行中执行启动程序

3、生成图片
