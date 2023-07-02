import pickle
import matplotlib.pyplot as plt
import numpy as np


def plot_trend(x, y, std, opt):
    plt.plot(x, y)
    plt.xlabel('Sample Size')
    plt.ylabel('Accuracy')
    plt.title('Search Curve')
    plt.axhline(y=opt, color='r', linestyle='--', label='Optimal')
    plt.fill_between(x, np.array(y) - std, np.array(y) + std, alpha=0.5, color='lightblue')
    plt.legend()
    plt.show()


def plot_trend_2(x1, y1, std1, x2, y2, std2):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # 绘制第一组数据
    ax1.plot(x1, y1, 'b-', label='Val.')
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Val. Acc', color='b')
    ax1.tick_params('y', colors='b')

    # 设置第一组数据的y轴范围
    min_y1 = min(y1)
    max_y1 = max(y1)
    ax1.set_ylim(min_y1 - 0.001, max_y1 + 0.001)

    # 绘制第二组数据
    ax2.plot(x2, y2, 'r-', label='Test')
    ax2.set_ylabel('Test Acc', color='r')
    ax2.tick_params('y', colors='r')

    # 设置第二组数据的y轴范围
    min_y2 = min(y2)
    max_y2 = max(y2)
    ax2.set_ylim(min_y2 - 0.001, max_y2 + 0.001)

    # 绘制标准差范围
    ax1.fill_between(x1, np.array(y1) - std1, np.array(y1) + std1, alpha=0.5, color='lightblue')
    ax2.fill_between(x2, np.array(y2) - std2, np.array(y2) + std2, alpha=0.5, color='lightpink')

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Search Curve')
    plt.show()


def plot_search_curve(filename, start=105, end=405, opt=0.91606):
    with open(file=filename, mode='rb') as f:
        datas = pickle.load(f)

    q_to_v_valid = {i: [0] * 10 for i in range(start, end, 5)}
    q_to_v_test = {i: [0] * 10 for i in range(start, end, 5)}
    for seed, data in enumerate(datas):
        for q_acc_dict in data['valid']:
            for x, y in q_acc_dict.items():
                for q, v in q_to_v_valid.items():
                    if x <= q:
                        q_to_v_valid[q][seed] = max(q_to_v_valid[q][seed], max(y))
        for q_acc_dict in data['test']:
            for x, y in q_acc_dict.items():
                for q, v in q_to_v_test.items():
                    if x <= q:
                        q_to_v_test[q][seed] = max(q_to_v_test[q][seed], max(y))


    x_valid = []
    y_valid = []
    std_valid = []

    x_test = []
    y_test = []
    std_test = []

    opt_key = 0
    for key, value in q_to_v_valid.items():
        if key % 50 == 0 or key % 190 == 0:
            print(key, np.mean(value), np.std(value))
        if np.mean(value) >= opt and opt_key == 0:
            opt_key = key
            print(key, np.mean(value) * 100, np.std(value) * 100)
        x_valid.append(key)
        y_valid.append(np.mean(value))
        std_valid.append(np.std(value))

    for key, value in q_to_v_test.items():
        #print(key, np.mean(value), np.std(value))
        if key % 50 == 0 or key % 190 == 0 or key == opt_key:
            print(key, np.mean(value), np.std(value))
        x_test.append(key)
        y_test.append(np.mean(value))
        std_test.append(np.std(value))



    plot_trend(x_valid, y_valid, std_valid, opt)


    plot_trend_2(x_valid, y_valid, std_valid, x_test, y_test, std_test)


if __name__ == '__main__':
    plot_search_curve('/home/gary/Desktop/surrogate-201/20230530-020241nb101_experiments_30_50_192_4_4_256/nb101_30_50_192_record.pkl',
                      start=81, end=195, opt=0.9505)
