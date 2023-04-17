import numpy as np

data = np.loadtxt('has_title.txt', encoding='utf-8', dtype='str_')
print(data)
a = np.array([1, 2, 3, 4])
print(a)
print(a.astype(dtype='float_'))
# 布尔型
c = np.array([0, 1, 2, 3, 4], dtype=np.bool_)
print(c)
# 字符串
d = np.array([0, 1, 2, 3, 4], dtype=np.str_)
print(d)
f = np.array([0, 1, 2, 3, 4], dtype=np.string_)
print(d,f,d.dtype,f.dtype)
s = np.array(['你好', 0, 1, 2], dtype=np.str_)
print(s.dtype)
# 创建结构化数据类型
dt = np.dtype([('age', 'U3')])
print(dt)
obj = np.array([('你们'), 234], dtype=dt)
print(obj)
print(obj['age'])
print('数据类型', obj.dtype)
"""
学生 三个特征: 姓名 年龄 成绩 定义结构化数据类型
str 字段 name
int 字段 age
float 字段 score
"""
# 定义数据类型
dt = np.dtype([('name', 'a', 3), ('age', 'i2'), ('score','f4')])
stu = np.array([('ly', 18, 89.894), ('wmd', 19, 20.23)], dtype=dt)
print(stu)
print('名字', stu['name'])
print('年龄', stu['age'])
print('成绩', stu['score'])
# 读取csv文件
data = np.loadtxt('wb.csv',dtype=np.str_, delimiter=',')
print(data)

# 数据读取
# 自定以数据类型
def parse_data(num):
    if num:
        return int(num)
    else:
        return 0
dt = np.dtype([('age', 'i1'), ('height', 'i2')])
print(dt)
stu = np.loadtxt('has_empty_data.csv', dtype=dt, skiprows=1, delimiter=',', usecols=(1, 3),
                 converters={1:parse_data,3:parse_data})
print(stu)
print(stu['age'], stu['age'].dtype)
print(stu['height'], stu['age'].dtype)
# 根据显示情况,学生不会为0，为了减少误差，可以使用中位数代替
# 中位数
ages = stu['age']
ages[ages==0] = np.median(ages)
print(ages)

a = np.array([2,0,1,5,8,3])
print(a)
print(type(a))
print(a.max())  # 最大值
print(a.min())  # 最小值
# 索引取值
print(a[3])
# 切片取值
print(a[1:4])
# 排序
a.sort()
print(a)
print(a.shape)  # 打印形状(6,)----6为行索引，没有列
b = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print(b)
print(type(b))  # <class 'numpy.ndarray'>
print(b.shape)
# 索引取值
print(b[0][2])
# 切片操作
print(b[2][:])
print(b[:][0])
print(b[0][:])
print(b[:][2])
# 获取列的值
print(b[:,1])  # 获取列所有第二行
print(b[1,:])  # 获取行所有第二列
print(b[:, np.newaxis, 3])  # 获取列得原本形状
"""
数组(arrays) 三维
特点: 1.array()吧python的类型转换成np的ndarray类型
   2.转换过数组的原始数据不变
函数：1.最小值 min()
     2.最大值 max()
     3.形状 shape属性 输出结果：(6,) 一维6长度
    索引和切片
         1.通过元素的下标位
         2.通过元素的 start:stop
         排序 
             1.sort
             2.sorted
    获取列的值
         1.[1,1] 返回一个一维数组
         2.[1,np.newaxis,1]  保留数据原本的位置返回数组
"""
c = np.array([[[1,2,3],[4,5,6],[7,8,9]]])
print(c)
print(c[0][0][0])
print(c[:][:][0])
print(c[:][0][1])
"""
创建数组arange() 默认:一维数组
更多维度reshape() 传入一个元组(,,,)可以定制多维度的形状
"""
a = np.arange(20)  # 滴鼻range()方法----->一维
print(a)
print(type(a)) # <class 'numpy.ndarray'>
b = np.arange(20).reshape((5,4))
print(b)
c = np.arange(20).reshape(5,4)
print(c)
"""
参数 元组 shape
    创建数组0填形状的多维数组zeros()
    默认类型:float
    dtype 可以设置元素类型 np.uint8 参数
    astype 重置数组元素
    创建数组1填充形状的多维数组ones()
"""
a = np.zeros((3,3), dtype=np.uint8) # 生成一个全为0的ndarray,元素类型是整形
print(a)
b = a.astype(np.float64)  # 浮点型数据
print(b)
# 生成全为1的整数
c = np.ones((4,4),dtype=np.uint8)
print(c)
"""
repeat() 重复元素 1，元素 2.重复次数
"""
a = np.repeat((1, 2, 3, 4), 5)
print(a)
print(type(a))  # <class 'numpy.ndarray'>
"""
广播:广播是一种强大的机制，它允许numpy在执行算数时使用不同形状的数组
"""
# a 和 b shape 相同
a = np.array([[1,2,3],[4,5,6]])
b = np.array([[10,10,10],[20,20,20]])
c = a*b
print(c)
print(a.shape, b.shape)
print('******')
# 形状不相同时
num1 = np.array([[1,2,3],[10,10,10],[20,20,20]])
num2 = np.array([1,2,3])
num3 = num1+num2
print(num3)
print(num1*num2)
"""
数学函数 包含了大量各种数学运算得函数，三角函数，算数函数，复数处理等
三角函数 sin()正弦  cos()余弦 tan()正切
反三角函数 arcsin()反正弦 arccps()反余弦 arctan()反正切
反转计算成角度 degrees(sin,cos,tan) 返回角度值
"""
a = np.array([30, 45, 60])
# 弧度公式 np,pi / 180
sin_num = np.sin(a * np.pi / 180)
cos_num = np.cos(a * np.pi / 180)
tan_num = np.tan(a * np.pi / 180)
print(sin_num)
print(cos_num)
print(tan_num)
print('正弦', sin_num)  # sin30 = 1/2
print('余弦', cos_num)  # cos60 = 1/2
print('正切', tan_num)  # tan45 = 1

sin_ang = np.arcsin(sin_num)  # 反正弦
cos_ang = np.arccos(cos_num)  # 反余弦
tan_ang = np.arctan(tan_num)  # 反正切
print('反正切', sin_ang)
print('反余弦', cos_ang)
print('反正切', tan_ang)
print('反转正弦角度', np.degrees(sin_ang))
print('反转余弦角度', np.degrees(cos_ang))
print('反转正切角度', np.degrees(tan_ang))
"""
舍入函数
around函数 四舍五入值 1.数组 2.舍入的小数位数:默认为0
floor 函数 向下取整数
ceil 函数 向上取整数
"""
a = np.array([20.3, 23.4, 45.5, 35.9,])
print(a)
print('四舍五入', np.around(a))
print('向下取整数', np.floor(a))
print('向上取整数', np.ceil(a))
"""
算法函数 数组的加减乘除，数组必须具有相同的形状，符合广播的规则
add() 加法
subtract()减法
multiply()乘法
divide() 除法
reciprocal() 倒数，元素的倒数1/4倒数4/1
power() 1.底数 2、幂次方
mod() 两数组余数 
"""
a = np.arange(9, dtype=np.float64).reshape(3,3)
b = np.array([10,10,10])
print(a)
print(b)
print('****')
print('add相加', np.add(a, b))
print('subtract相减',np.subtract(a, b))
print('multiply相乘', np.multiply(a, b))
print('divide相除', np.divide(a, b))
c = np.array([0.25, 1.33, 1, 100])
print('倒数reciprocal', np.reciprocal(c))
d = np.array([10,100,1000])
print('幂次方', np.power(d, 3))
# 两数组余数
num1 = np.array([10,20,30])
num2 = np.array([3,5,7])
print('两数组余数', np.mod(num1,num2))
"""
统计函数 从数组中快速查询最小，最大，百分位标准差,方差等

amin()参数：最小值 axis 1行比， 0列比
amax()参数: 最大值 axis 1行比，0列比
ptp(): 最大值-最小值的差
percentile(): 量度，百分比，1，数组，2，半分比0-100 
median(): 计算数组的（中位数） 中值
mean(): 算数平均值
std(): 标准差sqrt(mean(x-x。mean())**2)
标准差是一组数组的平均值分散成度的一种度量
标准差应用于投资上，可作为量度回报稳定性的指标。标准差数值越大，代表回报远离过去平均数值，回报较不稳定故风险越高。相反，标准差数值越小，代表回报较为稳定，风险亦较小。
var() 方差 mean((x-x, mean())**2), 平均数之差的平方的平均数
衡量随机变量或一组数据时离散程度的量度
    衡量随机变量或一组数据时离散程度的度量
    求和 ndarray.sum()
    axis=0,从上往下查找:',m1.sum(axis=0)
    axis=1,从左往右查找',m1.sum(axis=1)
    
    加权平均值 numpy.average()
    即将各数值乘以相应的权数，然后加总求和得到总体值，再除以总的单位数
    numpy.average(a, axis=None, weights=None, returned=False)
    - weights： 数组，可选

`与 a 中的值关联的权重数组。 
a 中的每个值都根据其关联的权重对平均值做出贡献。
权重数组可以是一维的(在这种情况下，它的长度必须是沿给定轴的 a 的大小)或与 a 具有相同的形状。
如果 weights=None，则假定 a 中的所有数据的权重等于 1。
一维计算是： avg = sum(a * weights) / sum(weights)

对权重的唯一限制是 sum(weights) 不能为 0。`
"""
a = np.array([[3,7,5],[8,4,3],[2,4,9]])
print(a)
print('行最大值', np.amax(a, axis=1))
print('列最大值', np.amax(a, axis=0))
print('行最小值', np.amin(a, axis=1))
print('行最大值', np.amin(a, axis=0))

print('最大值-最小值', np.ptp(a))
print('行最大-行最小', np.ptp(a, axis=1))
print('列最大-行最小', np.ptp(a, axis=0))
# 50%的分位数，a排序后的中位数
print('百分值', np.percentile(a,50))
print('行百分值', np.percentile(a, 50,axis=1))
print('列百分值', np.percentile(a, 50,axis=0))

# 中位数
print('中位数', np.median(a))
print('行中位数', np.median(a, axis=1))
print('列中位数', np.median(a, axis=0))











