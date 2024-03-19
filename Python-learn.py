# 注释
a = 1
"hello world"
str_v = "hello world" # 字符串
num_v = 45625 # 数字 整型或浮点型
bool_v = True # 布尔型bool
set_v = {1,2,3,4,5} # 集合set
tuple_v=(1,2,3,4) # 元组 tuple
list_v=[1,2,3] #列表 list
dict_v={'a':1,'b':2,'c':3} #字典 dict
# Python 语言的标准输出方法是：print(变量名)。print 函数的原
# 理是输出仅一个单独的变量
print(str_v)
print(num_v)
print(bool_v)
print(set_v)
print(tuple_v)
print(list_v)
print(dict_v)
# 字符串 字符串用引号括起来，双引号和单引号都可以
# 当字符串变量内含有单引号，就用双引号来表示该字符串；
# 当字符串变量内含有双引号，就用单引号来表示该字符串；
str1='"The cat catches the mat"'
str2="'The cat catches the mat'"
str3="""The cat catches the mat"""
print(str1)
print(str2)
print(str3)
# 想在字符串中插入其它变量，可使用“f 字符串”的方法
str4=f"hello{str1},{str3}"
print(str4)
answer = 0.98
print(f"测试集的准确率为: {answer}")
# 输出语句的转义字符
# 在字符串中添加转义字符，如换行符\n 与制表符\t，可增加 print 的可读性。
# 同时，转义字符只有在 print 里才能生效，单独对字符串使用无效。
message="\tint\n\tfloat\n\tbool\n\tstring\n\tset\n\ttuple\n\tdict\n\tlist"
print(message)
# 转义字符只有在print中才能 生效
# 运算符
a=1
b=5
c=2
print(a+b) # 加法
print(a-b) # 减法
print(a*b) # 乘法
print(a/b) # 除法
print(a**b) # 幂运算
print(a//b) # 取整
print(a%b) # 取余
# 布尔型只有两个值（True or False），通常是基于其他变量类型来进行生成
# ⚫对字符串作比较，使用等于号==与不等号!=；
# ⚫ 对数字作比较，使用大于>、大于等于>=、等于==、小于<、小于等于<=。
# （1）基于基本变量类型生成
str5="cxc"
print(str5=="cec")
print(str5!="cec")
num1=15
print(num1!=15)
print(num1==12)
print(num1>12)
print(num1<=15)
# （2）基于高级变量类型生成
# 集合
set1_v={1,2,3,4}
print(1 in set1_v)
print(2 not in set1_v)
#元组
tuple1_v=(1,2,3,4)
print(1 in tuple1_v)
print(1 not in tuple1_v)
#列表
list1_v=[1,2,3,4,5]
print(1 in list1_v)
print(1 not in list1_v)
#字典
dick1_v={'a':1,'b':2}
print(1 in dick1_v)
print( 1 not in dick1_v)
# （3）同时检查多个条件
# and 的规则是，两边全为 True 则为 True，其它情况均为 False；
# or 的规则是，两边有一个是 True 则为 True，其他情况为 False。
T = True
F = False
print(T and F)
print(T or F)
print(not True)
# 判断语句
if(1 == 2):
    print(1==2)
elif(1==3):
    print(1==3)
else:
    print("1==1")
"""
bool1 = False
bool2 = False
bool3 = False
if bool1:
 print('当 bool1 为 True，此行将被执行')
elif bool2:
 print('否则的话，当 bool2 为 True，此行将被执行')
elif bool3:
 print('否则的话，当 bool3 为 True，此行将被执行')
else:
 print('否则的话，此行将被执行')
"""
# 基本变量间的转换
str_1='123'
int_1=123
float_1=13.332
bool_1=True
print(str(int_1),str(float_1),str(bool_1))
print(int(str_1),int(float_1),int(bool_1))
print(float(str_1),float(int_1),float(bool_1))
print(bool(str_1),bool(float_1),bool(int_1))
# 注：其它变量转为布尔型变量时，只有当字符串为空、数字为 0、集合为空、
# 元组为空、列表为空、字典为空时，结果才为 False。
#三，高级变量类型
# 集合
set([1,2,3,4]) # 第一种通过set函数将列表转化为函数
{'a','b','c','d'} # 第二种直接利用大括号进行创建
# 元组
(1,2,3,4) # 直接小括号进行创建
1,2,3 # 省略括号直接创建
# （2）输出语句中的元组法
# 一个释放自我的元组
'a', 1, True, {1,2,3},(1,2,3),[1,2,3],{'a':1, 'b':2, 'c':3}
# 元组法替代 f 字符串
anwser=98
print(f"最终的答案为： {anwser}")
print("最终的答案为：",anwser)
# 元组法输出相对于f 字符串输出有个缺点，即输出的元素之间含有一个空格。
# （3）元组拆分法
# 迅速创建变量
a,b,c=1,2,3
print(c,b,a)
# 迅速交换变量值
a,b=b,a
print(a,b)
# 只要前两个答案
values=99,98,97,96,95,94
a,b,*rest=values
print(a,b)
print(rest)
# 列表
#创建列表
list1=['hello','world','!!!']
list2=[11,25,48,96]
list3=['hello',15,{15,15,16},(1,2,3),{'a':1,'b':2}] # 一个释放自我的列表
print(list1)
print(list1[0]) # 访问列表第一个元素
print(list1[1]) # 访问列表第二个元素
print(list1[-1]) # 访问列表导数第一个元素
print(list1[-2]) # 访问列表倒数第二个元素
list1[1]=5
print(list1)
# 切片
print(list3)
print(list3[1:4])  # 从索引[1]开始，切到索引[4]之前
print(list3[ : 4]) # 从列表开头开始，切到索引[4]之前
print(list3[ 1 : ])# 从索引[1]开始，切到结尾
print(list3[2: ])  # 切除开头两个
print(list3[ : -2])# 切除结尾两个
print(list3[2:-2]) # 切除开头 2 个和结尾 2 个
print(list3[ : : 2])#每两个元素采集一次
print(list3[ : : 3])#每三个元素采集一次
print(list3[1:4:2])#切掉开头和结尾后每两个采集一次
# 创建 list_v 的切片 cut_v
list_v = [1, 2, 3]
cut_v = list_v[ 1 : ]
print(cut_v)
# 修改 cut_v 的元素
cut_v[1] = 'a'
print(cut_v)
# 输出 list_v，其不受切片影响
print(list_v)
# 字典
dick1={'a':56,'b':62,'c':58} # 字典可以理解为升级版的列表，每个元素的索引都可以自己定
print(dick1['a'])
# 字典元素的添加与删除
CSU={'华中科技大学':985,'武汉大学':985}
print(CSU)
CSU['华中科技学院']="大专"
print(CSU)
del CSU['华中科技学院']
print(CSU)
# 循环语句
# for循环遍历；列表
schools=['中南大学','华中科技大学','武汉大学']
for school in schools:
    print(f"{school}是个好大学")
print("I can't wait to visit you")
# for循环遍历字典
# 遍历索引字
schools={'华中科技大学':'光电','武汉大学':'化学','军械士官学校':'军事'}
for k in schools.keys():
    print(k)
for v in schools.values():
    print(v)
for k,v in schools.items():
    print(k,"好专业是",v)
# while循环
a = 1
while a<=5 :
    print(a)
    a+=1
# continue与break
# continue 用于中断本轮循环并进入下一轮循环，在 for 和 while 中均有效。
# break 用于停止循环，跳出循环后运行后续代码，在 for 和 while 中均有效。
# break演示
a = 1
while True :
    if a==3 :
        break
    print(a)
    a+=1
a = 1
while a<5 :
    a+=1
    if a==3:
        continue
    print(a)
# 列表推导式
# 求平方——循环
value=[]
for i in [1,2,3,4,5]:
    value=value+[i**2]
print(value)
# 求平方——列表推导式
value = [i**2 for i in [1,2,3,4,5]]
print(value)
# 高级变量间的转换
"""
集合、元组、列表、字典四者之间可以无缝切换，需要用到四个函数
⚫ 转换为集合使用 set 函数；
⚫ 转换为元组使用 tuple 函数；
⚫ 转换为列表使用 list 函数；
⚫ 转换为字典使用 dict 函数。
"""
set_v = {1,2,3}
tuple_v = (1,2,3)
list_v = [1,2,3]
dict_v = { 'a':1 , 'b':2 , 'c':3 }
# 转化为集合
print( set( tuple_v ) )
print( set( list_v ) )
print( set( dict_v.keys() ) )
print( set( dict_v.values() ) )
print( set( dict_v.items() ) )
# 转化为元组
print( tuple( set_v ) )
print( tuple( list_v ) )
print( tuple( dict_v.keys() ) )
print( tuple( dict_v.values() ) )
print( tuple( dict_v.items() ) )
# 转化为列表
print( list( set_v ) )
print( list( tuple_v ) )
print( list( dict_v.keys() ) )
print( list( dict_v.values() ) )
print( list( dict_v.items() ) )
# 转化为字典
print(dict(zip({'a','b','c'},set_v)))
print(dict(zip({'a','b','c'},tuple_v)))
print(dict(zip({'a','b','c'},list_v)))
# 注：在使用dict函数时需要搭配zip函数，zip函数是将两个容器内元素进行配对
# 四。函数
"""
函数可以避免大段的重复代码，其格式为
def 函数名(输入参数):
 ''' 文档字符串 '''
 函数体
 return 输出参数
 文档字符串用于解释函数的作用，查看某函数文档字符串的方法是.__doc__。
第四行的 return 可省略（一般的函数不会省略），若省略，则返回 None。
"""
# 吞吐各个类型的变量
def my_fuck(v):
    '''我的函数'''
    return v
str_v=my_fuck("cxc") # 修改输入参数变量类型
print(str_v)
"""
str_v = my_func( "cxk" ) # 字符串
str_v
Out [2] : 'cxk'
In [3] : num_v = my_func( 123 ) # 数字
num_v
Out [3] : 123
In [4] : bool_v = my_func( True ) # 布尔型
bool_v
Out [4] : True
In [5] : set_v = my_func( {1,2,3} ) # 集合
set_v
Out [5] : {1, 2, 3}
In [6] : tuple_v = my_func( (1,2,3) ) # 元组
tuple_v
Out [6] : (1, 2, 3)
In [7] : list_v = my_func( [1,2,3] ) # 列表
list_v
Out [7] : [1, 2, 3]
In [8] : dict_v = my_func( {'a':1, 'b':2 , 'c':3 } ) # 字典
dict_v
"""
# 函数内部的空间是独立的，函数内部的变量叫做形式参数，不影响外界的实
# 际参数。在刚刚的例子中，In [1]函数体内的 v 就是形式参数，它在外界的实际空
# 间中是不存在的，只在调用函数的过程中会在函数空间内临时存在
# 吞吐多个变量
#吞吐多个普通参数
def my_counter(a,b):
    '''加法器和乘法器'''
    return a+b,a*b
(x,y)=my_counter(5,6)
print((x,y))
# 吞吐一个任意参量的参数
def menu(*arg):
    '''菜单'''
    return arg
info=menu('蛋炒饭','鱼香肉丝','牛肉汤')
print(menu)
# 吞吐多个普通参数和一个任意数量的参数
def her_hobbies(name,*hobbies):
    return name,hobbies
n,h=her_hobbies('jack','sing','playing')
print(f"{n},{h}")
# 吞吐多个普通参数，并附带一个任意数量的键值
def evaluate(in1,in2,**kwargs):
    ''' 先对计算机类评价，再对通信类评价，也可自行补充 '''
    kwargs['计算机类']=in1
    kwargs['通信工程']=in2
    return kwargs
eva1=evaluate('打代码的','拉网线的')
print(eva1)
# 额外补充法
eva2 = evaluate(
 '打代码的' ,
 '拉网线的' ,
 电子工程 = '焊电路的' ,
 能源动力 = '烧锅炉的'
)
print(eva2)
# 关键字调用
def my_evaluate1(college, major, evaluate):
   '''对某大学某专业的评价'''
   message = f"{college}的{major}{evaluate}。"
   return message
# 顺序调用
info = my_evaluate1('三峡大学', '电气工程', '挺厉害')
print(info)
# 关键字调用
info = my_evaluate1('三峡大学', evaluate='也还行', major='水利工程')
print(info)
# 输入函数的默认值
# 函数的默认值
def my_evaluate2(college, level='带专'):
    message = f"{college}, 你是一所不错的{level}!"
    return message
info = my_evaluate2('中南大学') # 遵循默认值
print(info)
info = my_evaluate2('铁道学院' , level='职高') # 打破默认值
print(info)
# 类
"""
⚫ 类的本质：在一堆函数之间传递参数；
⚫ 根据约定，类的名称需要首字母大写；
⚫ 类中的函数叫方法，一个类包含一个__init__方法 + 很多自定义方法，
__init__特殊方法前后均有两个下划线，每一个类中都必须包含此方法。
"""
# 示例
class Counter:
    def __init__(self,a,b):
        '''a,b是公共变量，也是self的属性'''
        self.a=a
        self.b=b
    def add(self):
        '''加法'''
        return self.a+self.b
    def sub(self):
        '''减法'''
        return self.a-self.b
cnt=Counter(5,6)
print(cnt.a,cnt.b)
print(cnt.add())
# 属性的默认值
# 前面讲过，属性即公共变量，上一个实例里的属性即 a 和 b。
# 可以给self 的属性一个默认值，此时默认值不用写进__init__后面的括号里。
# 带有默认值的参数
class Man:
    '''一个真正的 man'''

    def __init__(self, name, age):
        '''公共变量'''
        self.name = name
        self.age = age
        self.gender = 'man'  # 一个带有默认值的属性
    def zwjs(self):
        '''自我介绍'''
        return f"大家好！我是{self.name}，今年{self.age}岁了！"
# 创建与使用类
cxk = Man('鸡哥',24)
print(cxk.name, cxk.age) # 访问属性
print(cxk.zwjs())
print(cxk.gender)
# 修改默认值
cxk.gender='Neither man nor woman'
print(cxk.gender)
# 继承
"""
继承：在某个类（父类）的基础上添加几个方法，形成另一个类（子类）。
⚫ 父类从无到有去写属性和方法，第一行是 class 类名:
⚫ 子类可继承父类的属性和方法，第一行是 class 类名(父类名) :
子类在特殊方法里使用super()函数，就可以继承到父类的全部属性与方法。
"""
class Counter2:
    def __init__(self,a,b):
        '''引用父类的属性'''
        super().__init__(a,b) # 继承父类
    def mul(self):
        '''乘法'''
        return a*b
    def div(self):
        '''除法'''
        return a/b
test=Counter2(3,4)
print(test.sub())
print(test.mul())
"""
如果想要在子类中修改父类中的某个方法，可以直接在子类里写一个同名方
法，即可实现覆写，你也可以把覆写说成“变异”。
"""
# 掠夺
# 继承只能继承一个类的方法，但如果想要得到很多其它类的方法，则需要掠
# 夺功能。有了掠夺功能，一个类可以掠夺很多其它的类
# 掠夺者
class Amrc:
    def __init__(self,c,d):
        self.c=c
        self.d=d
        self.cnt=Counter(c,d)
    def mul(self):
        '''乘法'''
        return self.c * self.d
    def div(self):
        '''除法'''
        return self.c / self.d
test = Amrc(3,4) # 创建实例
print( test.mul() ) # 自己的方法
print( test.cnt.add() ) # 抢来的方法


























