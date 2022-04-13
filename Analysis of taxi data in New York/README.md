## 纽约计程车数据分析

*本文主要针对纽约计程车数据进行分析，旨在探索纽约居民的打车习惯，锻炼自己的数据分析能力。本文选取纽约市2016年1-6月的部分计程车数据，共计1458644行11列。
整个分析过程以`Python`语言作为工具，运用`Pandas`库处理数据清洗、缺省值、时序化数据等。运用`Seaborn`、`Matplotlib`进行数据可视化。*

***数据来源于:***[***kaggle***](https://www.kaggle.com/competitions/nyc-taxi-trip-duration)  
***纽约市出租车和轿车委员会发布的***[***历史数据集***](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

### 一、理解数据

* id－每次行程的唯一ID
* vendor_id－行程提供者的ID
* pickup_datetime－打表的日期和时间
* dropoff_datetime－停表的日期和时间
* passenger_count－车辆中的乘客数量
* pickup_longitude－上车的经度
* pickup_latitude－上车的纬度
* dropoff_longitude－下车经度
* dropoff_latitude－下车的纬度
* store_and_fwd_flag－行程记录是否为存储转发
* trip_duration－行程持续时间（s）

***基本认知：***　经纬度会为我们提供上下车的位置信息，同时我们可以根据上下车时间提取出month、week、day、hour等信息。行程持续时间可以帮助我们判断出长途、短途等信息。
乘客数量是帮我们判断乘客状态的信息，等等。行程提供者等信息本文暂不做分析。

### 二、提出问题
***首先：*** 我们假设本项目的目标是通过数据分析优化计程车的运营，给计程车司机的工作提供一定的指导。

***基于对数据的基本认知提出以下问题：***
* *何时为打车需求高发期*
* *夜间乘车情况*
* *每天乘客的结伴而行情况*
* *什么时间容易接到长途单*
* *每周的周几乘车需求比较高*
* *往返机场运营的合适时间点是什么时间*

### 三、数据整理（数据清理）


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
#忽略警告信息
warnings.filterwarnings('ignore')
#设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})
#读取数据
tx_data = pd.read_csv(r'F:\360MoveData\Users\Admin\Desktop\train.csv')
#数据类型转换：object -> datetime[ns]
tx_data['pickup_datetime'] = pd.to_datetime(tx_data['pickup_datetime'])
tx_data['dropoff_datetime'] = pd.to_datetime(tx_data['dropoff_datetime'])
#以'pickup_datetime'为基础，对数据进行扩展，抽取我们需要的数据，方便进行深入分析。
tx_data['pickup_yearmonth'] = tx_data.pickup_datetime.apply(lambda x: 100*x.year + x.month).astype(np.int64)
tx_data['pickup_month'] = tx_data['pickup_datetime'].dt.month
tx_data['pickup_week'] = tx_data['pickup_datetime'].dt.weekday
tx_data['pickup_day'] = tx_data['pickup_datetime'].dt.day
tx_data['pickup_hour'] = tx_data['pickup_datetime'].dt.hour
tx_data['pickup_date'] = tx_data.pickup_datetime.values.astype("datetime64[D]")
#查看数据类型
print(tx_data.dtypes)
#查看缺失值，因为数据发布前已经被处理过，所以我们不需要进行精细处理
print(tx_data.isna().any())
#将文件导出为`h5`文件，因为pd.read_hdf()的读取效率比pd.read_csv()要高。同时，h5文件不需要再次修正数据类型，方便我们分析时使用。
tx_data.to_hdf(r'F:\360MoveData\Users\Admin\Desktop\train.h5', key='df')
```

### 四、数据分析

```python
#读取数据
tx_data = pd.read_hdf(r'F:\360MoveData\Users\Admin\Desktop\train.h5', key='df')
#针对数据进行分类聚合处理
month_td = tx_data.groupby(['pickup_month'])["trip_duration"].agg(["sum", "mean", "count"])
month_td.reset_index(inplace=True)
day_td = tx_data.groupby(['pickup_day'])["trip_duration"].agg(["sum", "mean", "count"])
day_td.reset_index(inplace=True)
date_td = tx_data.groupby(['pickup_date'])["trip_duration"].agg(["sum", "mean", "count"])
date_td.reset_index(inplace=True)
#绘图
#针对month_td进行分析
fig = plt.figure(figsize=(20, 10), dpi=80)
axe1 = fig.add_subplot(121)
axe2 = fig.add_subplot(122)
axe1.plot(month_td['pickup_month'], month_td['count'], 'r.-')
axe1.set_xlabel('月份')
axe1.set_ylabel('车次')
axe1.set_title('每月打车次数')
axe2.plot(month_td['pickup_month'], month_td['mean'], 'm*-')
axe2.set_xlabel('月份')
axe2.set_ylabel('时间')
axe2.set_title('每月平均打车时间')
fig.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\month_td.png')
#针对date_td进行分析
fig = plt.figure(figsize=(20, 10), dpi=80)
axe = fig.add_subplot(111)
axe.plot(date_td['pickup_date'], date_td['count'], 'g*-')
axe.set_xlabel('天数')
axe.set_ylabel('车次')
axe.set_title('每月每天打车次数')
fig.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\date_td.png')
#针对day_td进行分析
fig = plt.figure(figsize=(20, 10), dpi=80)
axe = fig.add_subplot(111)
axe.plot(day_td['pickup_day'], day_td['count'] / 6, 'b+-')
axe.set_xlabel('天数')
axe.set_ylabel('车次')
axe.set_title('平均每天打车次数')
fig.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\day_td.png')

```
![month_td](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/month_td.png)
![date_td](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/date_td.png)
![day_td](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/day_td.png)

* 从趋势上看1月到六月份的打车时长是不断增加的，但是打车次数却从3月份开始下降，只有前仨月保持增长，可能用户逐渐习惯了更远距离也打车。
* 3-5月的订单量保持在较高的水准，6月份却明显降低，同时打车时长也明显放缓。
* 1-6月份的每日打车需求量基本保持在同一水平线上，上下震动只有一月份和五月份有异常值。
    * 提取出异常值日期:2016-1-24,2016-5-30
    * 2016-1-24 美国遭遇暴雪天气，影响出行，此消息百度可查。
    * 2016-5-30 这一天是美国阵亡将士纪念日，可能与此有关，因为并非物理因素。
* 30-31号的乘车需求量出现明显的降低，这里还有一些值得挖掘的信息。

#### 1、何时为打车需求高发期
***思路：通过对一天乘车需求量的统计，来确定打车需求高发期***
```python
#计算六个月内的每个时段打车数量的总值
week_trip = tx_data.groupby(["pickup_month",'pickup_week','pickup_day','pickup_hour'])["trip_duration"].agg(["mean","count"])
week_trip = week_trip.reset_index()
x1 = tx_data.groupby(["pickup_month",'pickup_week','pickup_day','pickup_hour'])["passenger_count"].agg(["mean"])
x1 = x1.reset_index()
week_trip["passenger_mean"] = x1["mean"]
#绘图
plt.figure(figsize=(16,6))
sns.swarmplot(x="pickup_hour", y="count", data=week_trip)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\passenger_hour_mean.png')
```

![passenger_hour_mean](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/passenger_hour_mean.png)

* 由图可知，在白天8-18点打车数量一直保持在高位。
* 6-8点打车需求激增，9-10有个小的下降趋势，然后一直保持到13点一直增加，14-16有一个小的降幅，16-18属于猛增阶段，然后缓慢下降。
* 18-23点打车需求一直保持在白天需求量之上，0点降到与早上七点相近，然后逐渐下降至低点。
* 所以最好的营业时间是16-23点,8-16点处于正常水平，13-14和8-9点在白天中占比最高，但是8-16点需求基本相差不大。

#### 2、夜间乘车情况
***思路：通过对一天乘车需求量的统计，来确定夜间乘车情况***
```python
#计算六个月内的每个时段打车数量的总值
hour_times = tx_data.groupby(['pickup_hour',"pickup_month"])["passenger_count"].agg(["mean","count"])
hour_times.reset_index(inplace=True)
#绘图
plt.figure(figsize=(16,6))
sns.swarmplot(x="pickup_hour", y="count", data=week_trip)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\passenger_hour_mean.png')
```

![hour_count](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/hour_count.png)

* 由分析1可知，18-23点打车需求一直保持在白天需求量之上，0点降到与早上七点相近。
* 零点以前的打车需求量，都是在白天以上的，夜晚活动人员增加。
* 结合分析1，我们可以得知，纽约市民的夜生活很丰富，凌晨1点前才会陆续归家。


#### 3、每天乘客的结伴而行情况
***思路：通过统计平均乘车人数来判断结伴而行的情况***
```python
#计算六个月内的每个时段每次打车的平均乘客数
hour_times = tx_data.groupby(['pickup_hour',"pickup_month"])["passenger_count"].agg(["mean","count"])
hour_times.reset_index(inplace=True)
#绘图
plt.figure(figsize=(10,8))
sns.stripplot(x='pickup_hour', y='mean',data=hour_times)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\hour_mean.png')


```
![hour_mean](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/hour_mean.png)

* 由上图可知，从六点开始，每辆车的平均乘车人数一直在上升，直到凌晨4点。
* 18-4点，平均乘客人数都在白天平均乘客数量以上，市民大多结伴出行。
* 4点以后呈断崖式下跌，6点后逐渐上升，在上班时间结伴而行的乘车需求小于夜间。

```python
#计算一月中每天乘客数。
day_pg = tx_data.groupby('pickup_day')['passenger_count'].agg(["sum", "mean", "count"])
day_pg.reset_index(inplace=True)
#绘图
fig = plt.figure(figsize=(20, 10), dpi=80)
axe = fig.add_subplot(111)
axe.plot(day_pg['pickup_day'],day_pg['mean'], 'c+-')
axe.set_xlabel('天数')
axe.set_ylabel('乘客人数')
axe.set_title('平均每次乘客人数')
#绘图
fig.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\day_pg.png')
fig2 = plt.figure(figsize=(20, 10), dpi=80)
axe2 = fig2.add_subplot(111)
axe2.plot(day_pg['pickup_day'],day_pg['sum']/6, 'c+-')
axe2.set_xlabel('天数')
axe2.set_ylabel('乘客人数')
axe2.set_title('平均每天乘客人数')
fig2.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\day_pg2.png')
#绘图
plt.figure(figsize=(10,8))
sns.boxenplot(x="pickup_week", y="passenger_mean", hue="pickup_month", data=week_trip)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\箱型图.png')
```

![day_pg](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/day_pg.png)
![day_pg2](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/day_pg2.png)
***引用上图***
![day_td](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/day_td.png)
![day_td](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/%E7%AE%B1%E5%9E%8B%E5%9B%BE.png)
* 由图我们可以看出，一个月中30号的，平均乘客数目最大超过1.7,7、11、29号有明显下降趋势。
* 一个月中平均乘客人数基本位置绕着1.66上下波动。
* 每天平均乘客总人数曲线与平均乘车量曲线基本持平，乘车量降低时，乘客有明显的结伴而行情况，例如4-5号。
* 由箱型图可知，周一到周五乘客结伴情况基本不变，在周末结伴情况出现明显增加，所以，在周六周日市民更倾向于结伴而行。
#### 4、什么时间容易接到长途单

***思路：目前只能通过对乘车时间的统计来确定乘车距离的远近，但是要注意排除堵车的风险，所以对白天数据的分析可信度不高***

```python
week_trip = tx_data.groupby(["pickup_month",'pickup_week','pickup_day','pickup_hour'])["trip_duration"].agg(["mean","count"])
week_trip = week_trip.reset_index()
x1 = tx_data.groupby(["pickup_month",'pickup_week','pickup_day','pickup_hour'])["passenger_count"].agg(["mean"])
x1 = x1.reset_index()
week_trip["passenger_mean"] = x1["mean"]
#初步分析查看数据是否有异常值
plt.figure(figsize=(20,10))
sns.boxenplot(x='pickup_hour',y = 'mean',data = week_trip)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\长途单.png')
#清除异常数据
plt.figure(figsize=(20,10))
y_values = week_trip['mean'][week_trip['mean'] < 4000]
sns.boxenplot(x='pickup_hour',y = y_values,data = week_trip)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\长途单.png')
```
***这张图表发现了几个异常值，打车平均时长超过了10000秒，超过3个小时，远远超过数据表中的其他平均时长。***
![长途单2](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/%E9%95%BF%E9%80%94%E5%8D%952.png)
***整理后的图表***
![长途单](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/%E9%95%BF%E9%80%94%E5%8D%95.png)

* 整个数据图表呈波浪形变化，并且可以看出1-5点出长途单是比较高的，虽然他们的四分位数比较低。
* 而对于八点之后的数据我们并不好判断，因为有堵车的因素在里面，需要更多的数据进行判断。
* 据此我们可以推测，1-5点是接长途单的耗时间，并且在夜间竞争的压力比较小。

#### 5、每周的周几乘车需求比较高
***思路：通过对数据进行分类聚合对每周的订单进行统计***
```python
week = tx_data.groupby(['pickup_week','pickup_month'])["trip_duration"].agg(["sum", "mean", "count"])
week.reset_index(inplace=True)
#绘图
plt.figure(figsize=(10,8))
sns.stripplot(x="pickup_week", y="count", hue="pickup_month",data=week,jitter=False)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\week_count.png')
week_trip = tx_data.groupby(["pickup_month",'pickup_week','pickup_day','pickup_hour'])["trip_duration"].agg(["mean","count"])
week_trip = week_trip.reset_index()
x1 = tx_data.groupby(["pickup_month",'pickup_week','pickup_day','pickup_hour'])["passenger_count"].agg(["mean"])
x1 = x1.reset_index()
week_trip["passenger_mean"] = x1["mean"]
#绘图
plt.figure(figsize=(10,8))
sns.swarmplot(x="pickup_week", y='count', hue="pickup_month", data=week_trip)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\week_trip.png')

```
![week_count](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/week_count.png)
![week_trip](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/week_trip.png)

* 由图可知，打车需求由周一到周五是逐渐增加的，周六到周日出现明显的降幅，原因之一在于周六日拼车的比较多。
* 周二到周六乘车需求是最高的，市民更愿意出行。
* 周日的乘车需求，分布在200-300次的时间较多，可能有待挖掘信息。
* 周六周日，一月份打车需求有几天非常的低，周三有一天比较低，可能有待挖掘的信息。

#### 6、往返机场运营的合适时间点是什么时间
***思路：通过上下车位置信息确定来往于肯尼迪机场的订单***

```python
#给一个纽约大致的范围，如果从数据里提取最大值最小值，会导致地图范围过大，绘图范围过小
NY_longitude = (-74.03, -73.75)
NY_latitude = (40.63, 40.85)
#绘图
fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
ax[0].scatter(tx_data['pickup_longitude'].values, tx_data['pickup_latitude'].values,
              color='blue', s=1, alpha=0.1)
ax[1].scatter(tx_data['dropoff_longitude'].values, tx_data['dropoff_latitude'].values,
              color='green', s=1, alpha=0.1)
ax[0].set_ylabel('latitude')
ax[0].set_xlabel('longitude')
ax[0].set_title('pickup')
ax[1].set_xlabel('longitude')
ax[1].set_ylabel('latitude')
ax[1].set_title('dropoff')
plt.ylim(NY_latitude)
plt.xlim(NY_longitude)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\location.png')
```
***数据分布***
![location](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/location.png)

*以肯尼迪机场为例*

![location](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/%E8%82%AF%E5%B0%BC%E8%BF%AA%E6%9C%BA%E5%9C%BA.png)

使用`google map`获取肯尼迪机场的经纬度区间：  
longitude：(-73.80884052712032,-73.7762248644104)  
latitude：(40.64130309036366,40.666111306195724)

```python
#选取下车点在肯尼迪机场的数据
df = tx_data[tx_data['dropoff_longitude'].between(-73.80884052712032,-73.7762248644104) & tx_data['dropoff_latitude'].between(40.64130309036366,40.666111306195724)]
plt.figure(figsize=(20,20),dpi=100)
plt.scatter(df['dropoff_longitude'].values, df['dropoff_latitude'].values,
              color='green', s=1, alpha=1)
plt.title('dropoff')
plt.rcParams.update({"font.size":20})
plt.tick_params(labelsize=20)
plt.ylabel('latitude',fontsize = 20)
plt.xlabel('longitude',fontsize = 20)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\airport_aim.png')
plt.figure(figsize=(20,20),dpi=100)
plt.scatter(df['pickup_longitude'].values, df['pickup_latitude'].values,
              color='blue', s=1, alpha=1)
plt.title('pickup')
plt.rcParams.update({"font.size":20})
plt.tick_params(labelsize=20)
plt.ylabel('latitude',fontsize = 20)
plt.xlabel('longitude',fontsize = 20)
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\airport_from.png')
df2 = df.groupby(['pickup_hour'])['pickup_longitude'].agg(['sum','count']).copy()
df2.reset_index(inplace =True)
print(len(df))
plt.figure(figsize=(20,20),dpi=100)
plt.bar(x=df2['pickup_hour'],height=df2['count'])
plt.title('pickup')
plt.savefig(r'F:\360MoveData\Users\Admin\Desktop\taxi_nyc\airport_from_time.png')
```

![airport_aim](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/airport_aim.png)
![airport_from](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/airport_from.png)
![airport_from_time](https://github.com/LHY-sudo/Data-analysis-exercise/blob/main/Analysis%20of%20taxi%20data%20in%20New%20York/img/airport_from_time.png)

* 图一反映的是下车点在肯尼迪机场的，下车人数主要集中在航站楼，也有一部分在U型路上下车，这一块大多是航空货运公司。
* 图二可以看出乘车来机场的主要在曼哈顿，帝国大厦附近的密度较高。
* 图三反映出一天里哪个时段乘车去机场的需求量，可以看出，4-6点需求量增加，6-10点下降到四点水平然后剧增到12点，达到全天最高点，然后逐渐下降。
* 需求量最高的是14点，4-19点都比较高，最好的工作时间是12-18点，5-6点需求量也比较高，但是没有前者大。
* 这块最好进行从机场返回的数据分析，争取来回都在乘车需求量比较大的阶段，这样才能确定最佳运营时间，本次不再进行分析，分析代码都是一致的。


### 五、总结
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;我们的数据分析就此结束，本文只是挖掘出了部分信息，虽然数据量不大，但还可以更加细致的挖掘出其他信息。例如，
可以根据经纬度判断那些位置更容易载到乘客，结合位置和乘车频率分析出经常乘车的公司或个人等。
