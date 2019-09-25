# coding=utf8
__author__ = 'Zhiyue.Wei'

import numpy as np
import pandas as pd
import os
import datetime
import time
import math
import pickle
from pyecharts import options as opts
from pyecharts.charts import Line, Grid

# 全局变量定义
pre_score = 0

# 输入一个ranko的csv 返回整理后的数据
def data_preprocessing(renko_path):
    renko_data = pd.read_csv(renko_path, engine='python')
    time_period = pd.read_csv("510050_30min_window.csv", engine='python')
    # 提取需要的数据 time 和close (renko数据只与 close 有关）
    tmp_df = renko_data[['Date Time', 'Close']]
    # 构建空的array
    renko_date_array = np.array([])
    renko_status_array = np.array([])
    renko_value_array = np.array([])
    renko_datetimeindex_array = np.array([])

    current_value = 0
    renko_brick = abs(tmp_df['Close'][0] - tmp_df['Close'][1]) # 每一个砖块的距离
    for index, row in tmp_df.iterrows():
        if index == 0:
            baseValueHigh = tmp_df["Close"][0]
            baseValueLow = tmp_df["Close"][0]
            continue

        # 计算当前时间所在的时间片index （这一段是什么意思？）找到当前的时间有多少个30分钟，然后取整之后再取回来？
        datetime_upper = datetime.datetime.fromtimestamp(math.ceil(time.mktime(datetime.datetime.strptime(row['Date Time'], '%Y-%m-%d %H:%M:%S').timetuple()) / (30 * 60)) * 30 * 60)
        #print("检测用:", datetime_upper) # 相当于再将这些零散的时间重新变成 datetime 的格式
        try:
            # 找到在存放观测时间的文件中的位置（此时 datetime_index 为在原时间文件中的行index）
            datetime_index = time_period.loc[pd.to_datetime(time_period["datetime"]) == datetime_upper].index[0]
            #print("位置:", datetime_index)
        except Exception as e:
            # 如果在存放时间的文件中不存在，那么说明该时间片异常
            print("时间异常",datetime_upper)



        if row["Close"] > baseValueHigh:
            direction = 1
            renko_num = int(round(abs(row["Close"] - baseValueHigh) / renko_brick))
        elif row["Close"] < baseValueLow:
            direction = -1
            renko_num = int(round(abs(row["Close"] - baseValueLow) / renko_brick))
        for i in range(renko_num):
            current_value += direction
            renko_status_array = np.append(renko_status_array, direction)
            renko_date_array = np.append(renko_date_array, row['Date Time'])
            renko_value_array = np.append(renko_value_array, current_value)
            # renko_datetimeindex_Array中存放的是当前的renko对应的是哪一个观测时间点（这个可以用来求时间的间隔）
            renko_datetimeindex_array = np.append(renko_datetimeindex_array, datetime_index)
        if direction == 1:
            baseValueHigh = row["Close"]
            baseValueLow = row["Close"] - renko_brick
        elif direction == -1:
            baseValueLow = row["Close"]
            baseValueHigh = row["Close"] + renko_brick

    tmp = pd.DataFrame({'Date Time': renko_date_array, 'Status': renko_status_array, "Value": renko_value_array, "Date Time Index": renko_datetimeindex_array})
    print(tmp)
    # 注意：tmp中存放数据的形式为，DateTime当前renko对应的时间；Status涨跌的情况，用1与-1表示；Value为累积涨跌状况； Date Time index为时间片的序号？
    return tmp

# 该函数用于在一个截面下按规则打分  输入刻画renko形状的 frame 输出分数
def get_renko_score(renko_frame, present_score_time, time_period):
    global pre_score
    pd.options.mode.chained_assignment = None

    # score_time 为当前观测的时间点
    score_time = datetime.datetime.strptime(present_score_time, '%Y-%m-%d %H:%M:%S')
    # 从该截面向前取数据
    front_data = renko_frame.loc[
        pd.to_datetime(renko_frame["Date Time"]) <= datetime.datetime.strptime(str(score_time), '%Y-%m-%d %H:%M:%S')]
    tmp = front_data.iloc[-7:, :] #一开始在front_data中向前取10个

    # 当前的renko shape含有的结构为：Date Time, Status, Value, Date Time Index
    # 首先找到初始状态下的趋势：找到该观测区间内的最大值与最小值
    # 需要改进：比如，改进成当前区间内的离观测点最近的最大or最小值
    global pre_score
    renko_shape = tmp
    renko_num = len(renko_shape)
    if renko_num < 7:
        #print("The renko number is too small for the calculation")
        score=0
    else:
        # 找到观测的时间点在存放时间点的表中的位置
        scoreTimeIndex = time_period.loc[pd.to_datetime(time_period["datetime"]) == score_time].index[0]
        # 观测窗口开始的时间索引
        startIndex = renko_shape.index[0]
        # 寻找该时间窗口内的趋势（最大值与最小值）
        # 有关这里的趋势完整性的寻找有待改善
        temp_max = renko_shape['Value'].values[0]
        temp_min = renko_shape['Value'].values[0]
        for j in range(len(renko_shape['Value'].values)):
            if renko_shape['Value'].values[j] >= temp_max:
                temp_max = renko_shape['Value'].values[j]
                maxIndex = renko_shape['Value'].index[j]
                maxDatetimeIndex = renko_shape['Date Time Index'].values[j]
            if renko_shape['Value'].values[j] <= temp_min:
                temp_min = renko_shape['Value'].values[j]
                minIndex = renko_shape['Value'].index[j]
                minDatetimeIndex = renko_shape['Date Time Index'].values[j]
        maxValue = temp_max
        minValue = temp_min
        # 在找到了初始的序列的最大值与最小值之后，需要进一步进行趋势完整性的判断
        if maxIndex > minIndex:
            # 如果当前的趋势为递增，需要检查与该区间相邻的地方是否仍有上升趋势的延续
            if (minIndex-startIndex)<1:
                # 需要向前查找上升趋势开始的地方
                #print("需要更新上升")
                for i in range(len(front_data.iloc[:-11,:]['Date Time'])):
                    tmp_front_df=front_data.iloc[-11-i,:]
                    if tmp_front_df['Value']<minValue:
                        # 修正各项参数值：添加行、最小值、最小值的 index
                        target_index = front_data.iloc[:-11, :].index[-i - 1]
                        renko_shape.loc[target_index]=tmp_front_df
                        minValue=tmp_front_df['Value']
                        minIndex=front_data.iloc[:-11,:].index[-i-1]
                    else:
                        # 如果当前的趋势发生扭转，那么停止向前搜索
                        break
                renko_shape.sort_index(inplace=True)
                #print("The newly updated renko shape:", renko_shape)
        elif maxIndex < minIndex:
            # 同样地，需要对下跌趋势进行完整性判断
            if (maxIndex-startIndex)<1:
                #print("更新下降")
                for i in range(len(front_data.iloc[:-11,:]['Date Time'])):
                    tmp_front_df=front_data.iloc[-11-i,:]
                    if tmp_front_df['Value'] > maxValue:
                        # 修正各项参数的值：添加行、最大值、最大值的位置
                        target_index=front_data.iloc[:-11,:].index[-i-1]
                        renko_shape.loc[target_index]=tmp_front_df
                        maxValue=tmp_front_df['Value']
                        maxIndex=front_data.iloc[:-11,:].index[-i-1]
                    else:
                        # 如果趋势被打断，停止搜索
                        break
                renko_shape.sort_index(inplace=True)
                #print(renko_shape)
        print("检验renko shape:", renko_shape)


        # 重新更新开始的索引
        renko_num=len(renko_shape)
        # 窗口结束的时间更新
        lastTimeIndex=renko_shape.iloc[-1,:]['Date Time Index']


        # 打分的标准制定: 第一部为单纯对renko图像的形态进行打分
        # 然后倒着去找其震荡的情况;并且也需要获得观测窗口内部每一段趋势形成的时间
        # 提取出这个时间窗口内部status array
        status_array = []
        for element in renko_shape['Status'].values:
            status_array.append(element)
        print("检测用status_array:", status_array)

        # 将这个观测窗口里面的子列全部取出来
        # 需要记录每一个子列结束的时间
        status_subdict={}
        cumulative_subdict={}
        temp_array=[]
        for j in range(len(status_array)):
            current_status = status_array[j]
            if j==0:
                temp_array.extend([current_status])
                pre_status=current_status
                current_time=renko_shape.iloc[j,:]['Date Time Index']
            else:
                if current_status!=pre_status:
                    # 此时趋势发生反转
                    tmp_dict={'trend': temp_array, 'time': current_time}
                    status_subdict[j-1]=tmp_dict
                    temp_array=[]
                    temp_array.extend([current_status])
                else:
                    temp_array.extend([current_status])
                pre_status=current_status
                current_time=renko_shape.iloc[j,:]['Date Time Index']

        tmp_dict={'trend': temp_array, 'time': current_time}
        status_subdict[j]=tmp_dict
        # 状态分段数组：状态开始的位置；这个状态内部的变化情况
        print("检测状态分段数组:", status_subdict)

        key_array=[]
        for key in status_subdict:
            temp=status_subdict[key]
            #print(temp)
            cumulative_subdict[key]={'cumulative': np.sum(temp['trend']), 'time': temp['time']}
            key_array.extend([key])
        print("检测累积状态分段数组:", cumulative_subdict)

        count=0
        max_abs=0
        for value in cumulative_subdict.values():
            if abs(value['cumulative']) >= max_abs:
                direction=np.sign(value['cumulative'])
                max_abs=abs(value['cumulative'])
                max_position=count
                max_time=value['time']
            count = count + 1
        count=count-1
        trend=direction*max_abs
        print("定位trend:", trend, max_position, count, max_time)

        #print("key_array:", key_array, key_array[max_position])

        # 计算在这个窗口内部发生了多少次翻转
        fluctuation_num=0
        for keys in key_array:
            fluctuation_num=fluctuation_num+1
        print("当前窗口内部的波动次数为:", fluctuation_num)

        # 定位了最大的trend之后，还要看当前的位置与其之前以及之后的值有没有什么影响
        if max_position==count:
            print("趋势保持到末尾")
            #如果当前是最后的一个值，需要考虑：上涨或者下跌的幅度？（大于5就是满分）；在该位置之前的位置是否有值将其抵消？
            if max_position==0:
                print("只有连续涨或者跌***************************************************************")
                score=np.sign(trend)*7  #此时只有涨或者跌（这个时候不可以有慢慢上涨的情况，直接涨到最大）
            elif (abs(abs(cumulative_subdict[key_array[max_position-1]]['cumulative'])-abs(cumulative_subdict[key_array[max_position]]['cumulative']))<=1):
                #print(cumulative_subdict[key_array[max_position]], cumulative_subdict[key_array[max_position-1]])
                print("发生相似的反转")
                score=0
            else:
                # 没有发生很大的反转
                # 需要区分，如果这个观测窗口里的有很多次波动怎么办？
                if fluctuation_num>3 and abs(trend)<=3:
                    score=0
                else:
                    if abs(trend)>=4:
                        score=np.sign(trend)*7
                    else:
                        score=np.sign(trend)*(max(0, min(abs(trend)-1,7)))
        elif max_position < count:
            # 如果趋势在窗口的中间，那么需要考虑：后面有没有振动的值？后面振动的值的长度
            print("趋势在窗口的中间")
            count_after=count-max_position # 用于记录当前趋势后发生震荡的小区间段数目
            max_after=0 # 用于记录在趋势结束后一直到最后发展极值
            max_before=0
            for k in range(max_position+1, count+1, 1):
                if abs(cumulative_subdict[key_array[k]]['cumulative'])>max_after:
                    max_after=cumulative_subdict[key_array[k]]['cumulative']
            # 找到在趋势形成之前最大变动情况
            for k in range(0, max_position, 1):
                if abs(cumulative_subdict[key_array[k]]['cumulative'])>max_before:
                    max_before = cumulative_subdict[key_array[k]]['cumulative']
            print("max_after:", max_after, "max_before:", max_before, "count_after:", count_after)

            if (max_after*trend<0 and (abs(max_after)/abs(trend))>=0.5) or (max_before*trend<0 and (abs(max_before)/abs(trend))>=0.5):
                # 如果发生反转，且反转的程度超过原来的一半，则为0
                print("发生大小过半的反转")
                score=0
            elif max_after*trend<0 and 0.4<abs(max_after)/abs(trend)<0.5 :
                # 如果发生反转，但是反转的程度没有那么大
                print("发生略大的反转")
                score=np.sign(trend)*max(min(7, abs(trend)-max_after), 0)
            else:
                print("没有发生较大的反转")
                if abs(trend)>=5:
                    score=np.sign(trend)*7
                else:
                    score=np.sign(trend)*min(7, abs(trend)-count_after)

        print("没有考虑时间", score, "for", score_time)

        print("上一次的分数为:", pre_score)
        # 关于时间惩罚项
        # 趋势维持时间的惩罚（我现在只取趋势后三个格子的时间)
        trend_unit=12
        trend_len=abs(trend) #trend的长度
        print("trend_len:", trend_len, "trend_unit:", trend_unit)
        if trend_len > 3:
            trend_start=renko_shape.iloc[int(key_array[max_position] - 3 + 1), :]['Date Time Index']
            trend_end = renko_shape.iloc[key_array[max_position], :]['Date Time Index']
        else:
            trend_start= renko_shape.iloc[int(key_array[max_position] - trend_len + 1), :]['Date Time Index']
            trend_end = renko_shape.iloc[key_array[max_position], :]['Date Time Index']
        trend_period=math.floor((trend_end - trend_start) / trend_unit) #隔了多少天
        print("趋势间隔时间:", trend_period)
        # 考察趋势的延续时间是否大于4天
        if int(trend_period/4)>=1:
            print("当前的趋势大于4天")
            modi_score=0
        else:
            # 如果没有到达5天，那么我们可以认为为较大的趋势；并且对于观测时间与结束时间的间隔进行计算
            observe_span=math.floor((scoreTimeIndex-trend_end)/trend_unit)
            print("观测推移时间:", observe_span)
            if (observe_span==2 and pre_score>=5) or observe_span==1:
                modi_score=abs(score)
            else:
                modi_score = max(0, abs(score) - observe_span)


        if modi_score==0:
            score=0
        else:
            score=np.sign(score)*modi_score

        # 修改后的分数
        print("上一次的分数:", pre_score)
        print("修改后的分数:", score)
        pre_score=score
        #if score != 0:
            #print(score,"for",score_time)
            #print("当前时间的惩罚:", scoreTimeIndex, lastTimeIndex, scoreTimeIndex-lastTimeIndex)
    return score


# 该函数用于对截面打分进行循环 生成字典 key为时间 value为分数
def get_score(renko_frame):
    # 取出此表的起始和终止日期 用于确定打分时间的范围
    response = pd.read_csv("510050_30min_window.csv", engine='python')
    resArray = []

    # 尝试，只找到前面50个的
    for timeitem in response['datetime']:
        # 获取当前时间的renko形态并且给当前的形态打分
        renko_score = get_renko_score(renko_frame, timeitem, response)
        resArray.append([timeitem, renko_score])

    return resArray


def KlinePlotting(target_file):
    global trend_punishment
    res = pickle.load(open(target_file, 'rb'))
    c = (
        Line()
            .add_xaxis([item[0] for item in res])
            .add_yaxis(
            "打分",
            [item[1] for item in res],
            linestyle_opts=opts.LineStyleOpts(width=3, opacity=0.5),
        )
            .set_global_opts(
            title_opts=opts.TitleOpts(title="打分"),
            xaxis_opts=opts.AxisOpts(type_="category"),
            datazoom_opts=[
                opts.DataZoomOpts(
                    type_="slider",
                    xaxis_index=[0],
                    range_start=80,
                    range_end=100,
                )
            ],
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(245, 245, 245, 0.8)",
                border_width=1,
                border_color="#ccc",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
        )
    )

    gridChart = Grid(
        init_opts=opts.InitOpts(
            width="1000px",
            height="500px",
            animation_opts=opts.AnimationOpts(animation=False)
        )
    )
    gridChart.add(c, grid_opts=opts.GridOpts(pos_left="10%", pos_right="10%"))
    gridChart.render("score_modified_plot_new_0.002.html")


def init():
    # 获取数据
    brick_data = r'D:\\2019-2020Summer\\江海证券\\Task02_Renko\\RenkoScoring\\renkoCalc\\data\\'
    # 有五张表  可以输入0-4  是不同砖宽的数据
    renko = os.listdir(brick_data)[4]
    # 数据预处理 只保留有需要的数据
    renko_frame = data_preprocessing(brick_data + renko)
    # 生成打分字典
    score_array = get_score(renko_frame)
    # 将数据导入某一个文件
    pickle.dump(score_array, open('score2.pkl', 'wb'))
    # 在这里加上直接画图
    KlinePlotting('score2.pkl')
    return score_array


'''
说明：改程序用于对renko形态打分
输入：510051的本地数据的路径 在brick_data处修改
输出：打分的字典 key为打分的时间 value为当前时间的分数
'''

if __name__ == '__main__':
    # 510051的本地数据的路径
    init()
