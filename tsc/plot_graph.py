import matplotlib.pyplot as plt
import numpy as np
from collections import deque

import os
import sys
from tsc.eval_config import name
#将需要评估的文件加入到环境路径中
importPath = os.path.join(os.getcwd(), 'sumo_logs' + os.sep + 'eval_' + name)
sys.path.append(importPath)
#读取评估生成的文件画图
from avg_timeLoss import timeLoss
from avg_duration import duration
from avg_waitingTime import waitingTime

#为评估地图赋予名字
map_title = {
    'grid4x4': '4x4 Grid',
    'arterial4x4': '4x4 Avenues',
    'ingolstadt21': 'Ingolstadt Region',
    'cologne8': 'Cologne Region',
    # 'most_0': 'real net',
    'osm': 'nanshan',
    'base': 'fenglin',
    # 'rl_wj': 'rl_wj',
    'exp_0': 'large grid 0',
    # 'exp_1': 'large grid 1'
}


# compute best metric value
min_duration, min_timeLoss, min_waitingTime = 1e8, 1e8, 1e8
min_duration_index, min_timeLoss_index, min_waitingTime_index = None, None, None
try:
    for i in range(25):
        ans_duration = 0
        ans_timeLoss = 0
        ans_waitingTime = 0
        for map in ['base', 'grid4x4', 'arterial4x4', 'ingolstadt21', 'cologne8']:
            key_ = "{} {}".format(name, map)
            ans_duration += duration[key_][i]
            ans_timeLoss += timeLoss[key_][i]
            ans_waitingTime += waitingTime[key_][i]
        if ans_duration < min_duration:
            min_duration = ans_duration
            min_duration_index = i
        if ans_timeLoss < min_timeLoss:
            min_timeLoss = ans_timeLoss
            min_timeLoss_index = i
        if ans_waitingTime < min_waitingTime:
            min_waitingTime = ans_waitingTime
            min_waitingTime_index = i
except Exception as e:
    print(e)

print("Best duration index is: {}".format(min_duration_index))
for map in ['base', 'grid4x4', 'arterial4x4', 'ingolstadt21', 'cologne8']:
    key_ = "{} {}".format(name, map)
    print("{} duration: {}".format(map, duration[key_][min_duration_index]))

print("Best timeLoss index is: {}".format(min_timeLoss_index))
for map in ['base', 'grid4x4', 'arterial4x4', 'ingolstadt21', 'cologne8']:
    key_ = "{} {}".format(name, map)
    print("{} timeLoss: {}".format(map, timeLoss[key_][min_duration_index]))

print("Best waitingTime index is: {}".format(min_waitingTime_index))
for map in ['base', 'grid4x4', 'arterial4x4', 'ingolstadt21', 'cologne8']:
    key_ = "{} {}".format(name, map)
    print("{} waitingTime: {}".format(map, waitingTime[key_][min_duration_index]))


#需要评估的所有算法
alg_name = {
    'FIXED': 'Fixed Time',
    'STOCHASTIC': 'Random',
    'MAXWAVE': 'Greedy',
    'MAXPRESSURE': 'Max Pressure',
    'test': 'test'
}

#非自适应的算法作为baseline
statics = ['MAXPRESSURE', 'STOCHASTIC', 'MAXWAVE', 'FIXED']

num_n = -1
num_episodes = 20 #评估的区间长度
fs = 10
window_size = 5 #滑动平均的窗口

metrics = [timeLoss, duration, waitingTime]
metrics_str = ['Avg. Delay', 'Avg. Trip Time', 'Avg. Wait']

chart = {
    'test': {
        'Avg. Delay': [],
        'Avg. Wait': [],
        'Avg. Trip Time': []
    }
}

for met_i, metric in enumerate(metrics):
    print('\n', metrics_str[met_i])
    for map in map_title.keys():
        print()
        print(map_title[map])
        dqn_max = 0
        plt.gca().set_prop_cycle(None)
        for key in metric:
            if map in key and '_yerr' not in key:
                alg = key.split(' ')[0]
                key_map = key.split(' ')[1]

                if alg == 'IDQN': dqn_max = np.max(metric[key])     # Set ylim to DQN max, it's approx. random perf.

                if len(metric[key]) == 0:   # Fixed time isn't applicable to valid. scenario, skip color for consistency
                    plt.plot([], [])
                    plt.fill_between([], [], [])
                    continue

                # Print out performance metric
                err = metric.get(key + '_yerr')
                if num_n == -1:
                    last_n_ind = np.argmin(metric[key])
                    last_n = metric[key][last_n_ind]
                else:
                    last_n_ind = np.argmin(metric[key][-num_n:])
                    last_n = metric[key][-num_n:][last_n_ind]
                last_n_err = 0 if err is None else err[last_n_ind]
                avg_tot = np.mean(metric[key])
                avg_tot = np.round(avg_tot, 2)
                last_n = np.round(last_n, 2)
                last_n_err = np.round(last_n_err, 2)

                #last_n = np.round(np.mean(err), 2) if err is not None else 0
                #last_n = last_n_ind

                # Print stats
                if alg in statics:
                    print('{} {}'.format(alg_name[alg], avg_tot))
                    do_nothing = 0
                else:
                    print('{} {} +- {}'.format(alg_name[alg], last_n, last_n_err))
                    if not(map == 'grid4x4' or map == 'arterial4x4'):
                        chart[alg_name[alg]][metrics_str[met_i]].append(str(last_n)) #+' $\pm$ '+str(last_n_err)

                # Build plots
                if alg in statics:
                    plt.plot([avg_tot]*num_episodes, '--', label=alg_name[alg])
                    plt.fill_between([], [], [])      # Advance color cycle
                elif not('FMA2C' in alg or 'IPPO' in alg):
                    windowed = []
                    queue = deque(maxlen=window_size)
                    std_q = deque(maxlen=window_size)

                    windowed_yerr = []#计算滑动平均
                    x = []
                    for i, eps in enumerate(metric[key]):
                        x.append(i)
                        queue.append(eps)
                        windowed.append(np.mean(queue))
                        if err is not None:
                            std_q.append(err[i])
                            windowed_yerr.append(np.mean(std_q))

                    windowed = np.asarray(windowed)
                    if err is not None:
                        windowed_yerr = np.asarray(windowed_yerr)
                        low = windowed - windowed_yerr
                        high = windowed + windowed_yerr
                    else:
                        low = windowed
                        high = windowed

                    plt.plot(windowed, label=alg_name[alg])
                    plt.fill_between(x, low, high, alpha=0.4)
                else:
                    if alg == 'FMA2C':  # Skip pink in color cycle
                        plt.plot([], [])
                        plt.fill_between([], [], [])
                    alg = key.split(' ')[0]
                    x = [num_episodes-1, num_episodes]
                    y = [last_n]*2
                    plt.plot(x, y, label=alg_name[alg])
                    plt.fill_between([], [], [])  # Advance color cycle

        points = np.asarray([i for i in range(0, num_episodes, 2)])
        plt.yticks(fontsize=fs)
        plt.xticks(points, fontsize=fs)
        #plt.xlabel('Episode', fontsize=32)
        #plt.ylabel('Delay (s)', fontsize=32)
        plt.title(map_title[map] + '  ' + metrics_str[met_i], fontsize=fs)
        plt.legend(prop={'size': 10})
        bot, top = plt.ylim()
        if bot < 0: bot = 0
        #plt.ylim(bot, dqn_max)
        plt.show()

for alg in chart:
    print(alg)
    for met in metrics_str:
        print(met, ' & ', ' & '.join(chart[alg][met]), '\\\\')

