import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import math as m
import scipy.optimize as opt
import pandas as pd
import pypoman as pyp




data = pd.read_csv(r'C:\Users\Bruhonog\Desktop\six_sem\mat_stat\4_lab\Chanel\_1\_800nm\_0.03.csv', sep=';', encoding='cp1251')
eps = 0.76 * 10e-4
data = data['мВ']
interval_data = []
for i in range(0, len(data)):
    interval_data.append([data[i] - eps, data[i] + eps])

"""Визуализация данных выборки"""


def plot_data(data):
    plt.figure()
    data.plot(color='blue', linewidth=0.5)
    plt.title('Experiment data')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.show()


def diagram(data, epsilon, beta):
    plt.figure()
    plt.fill_between(data.index, data - epsilon, data + epsilon, color = 'lime', alpha = 0.3)
    plt.plot(data.index, data, color='cyan', linewidth=0.5)
    if beta is not None:
        plt.plot([0, 199],[beta, beta], color = "maroon", linestyle = "--", linewidth = 0.5)
    plt.title('Данные')
    plt.xlabel('n')
    plt.ylabel('mV')
    plt.show()

plot_data(data)
diagram(data, eps, None)


"""Варьирование неопределнности измерений. Без сужения интервалов"""


def minimization(A, y, eps, lim):
    [m, n] = A.shape

    c = np.concatenate((np.zeros((n, 1)), np.ones((m, 1))), axis=0)
    c = np.ravel(c)

    diag = np.diag(np.full(m, -eps))

    M_1 = np.concatenate((-A, diag), axis=1)
    M_2 = np.concatenate((A, diag), axis=1)
    M = np.concatenate((M_1, M_2), axis=0)

    v = np.concatenate((-y, y), axis=0)

    l_b = np.concatenate((np.full(n, None), np.full(m, lim)), axis=0)
    print(l_b)
    u_b = np.full(n + m, None)

    bounds = [(l_b[i], u_b[i]) for i in range(len(l_b))]

    result = opt.linprog(c=c, A_ub=M, b_ub=v, bounds=bounds)
    y = result.x

    coefs = y[0:n]
    w = y[n:n + m]

    return [coefs, w]


def parser(lim):
    data = pd.read_csv(r'C:\Users\Bruhonog\Desktop\six_sem\mat_stat\4_lab\Chanel\_1\_800nm\_0.03.csv', sep=';', encoding='cp1251')

    data_mv = np.ravel(data.drop('мА', axis=1).to_numpy())

    data_n = np.arange(1, len(data_mv) + 1, 1)

    data_eps = 1e-4

    data_X = np.stack((np.ones(len(data_mv)), data_n))
    data_X = np.transpose(data_X)
    [data_tau, data_w] = minimization(data_X, data_mv, data_eps, lim)

    with open(f'Chi{lim}.txt', 'w') as f:
        print(f'{data_tau[0]} {data_tau[1]}', file=f)
        for temp in data_w:
            print(temp, file=f)


def load_processed(filename):
    A = 0
    B = 0
    w = []
    with open(filename) as f:
        A, B = [float(t) for t in f.readline().split()]
        for line in f.readlines():
            w.append(float(line))
    return A, B, w


parser(1)
A1, B1, w1 = load_processed('Chi1.txt')
print([A1, B1])
print(np.sum(w1))
plt.fill_between(data.index, np.array(data) + np.array(w1) * eps, np.array(data) - np.array(w1) * eps, color = 'lime', alpha = 0.3)
plt.plot(np.arange(0, 200), A1 + B1 * (np.arange(0, 200)), label='lsm', color='maroon', linewidth = 0.5)
plt.xlabel('n')
plt.ylabel('мV')
plt.title('Варьирование неопределнности измерений. Без сужения интервалов')
plt.show()
plt.close()

"""С сужением интервалов"""


parser(0)
A0, B0, w0 = load_processed('Chi0.txt')
print(w0)
print([A0, B0])
print(np.sum(w0))
plt.fill_between(data.index, np.array(data) + np.array(w0) * eps, np.array(data) - np.array(w0) * eps, color = 'rosybrown', alpha = 0.7)
plt.fill_between(data.index, np.array(data) + eps, np.array(data) - eps, color = 'lime', alpha = 0.3)
plt.plot(np.arange(0, 200), A0 + B0 * (np.arange(0, 200)), label='lsm', color='maroon', linewidth = 0.5)
plt.xlabel('n')
plt.ylabel('мV')
plt.title('Варьирование неопределнности измерений. C сужением интервалов')
plt.show()
plt.close()


"""Векторы w1 и w0"""


plt.figure()
plt.plot(data.index, w0, linewidth = 0.5, label = 'w0')
plt.plot(data.index, w1, linewidth = 0.5, label = 'w1')
plt.legend()
plt.xlabel('n')
plt.ylabel('mV')
plt.title('Векторы w1 и w0')
plt.show()
plt.close()


"""Анализ регрессионных остатков. Модель без сужения интервалов"""


plt.figure()
plt.fill_between(data.index, np.array(data) + eps - (A1 + B1 * (np.arange(0, 200))), np.array(data) - eps - (A1 + B1 * (np.arange(0, 200))), color = 'blue', alpha = 0.3)
plt.plot([0, 199], [0, 0], label='lsm', color='maroon', linewidth = 0.5)
plt.xlabel('n')
plt.ylabel('мV')
plt.title('')
plt.show()
plt.close()


""""Модель с сужениями интервалов"""


plt.figure()
plt.fill_between(data.index, np.array(data) + eps - (A0 + B0 * (np.arange(0, 200))), np.array(data) - eps - (A0 + B0 * (np.arange(0, 200))), color = 'blue', alpha = 0.3)
plt.plot([0, 199], [0, 0], label='lsm', color='maroon', linewidth = 0.5)
plt.xlabel('n')
plt.ylabel('мV')
plt.title('')
plt.show()
plt.close()
