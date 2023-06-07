from functions import *


"""Частоты элементарных подинтервалов при вычислении моды модели"""


def subinterval_frequencies_and_mode(A, B, eps, data):
    y = []
    for i in range(0, len(data)):
        y.append(data[i] + eps - (A + B * i))
        y.append(data[i] - eps - (A + B * i))

    y = list(set(y))
    y.sort()

    z = []

    for i in range(0, len(y) - 1):
        z.append([y[i], y[i + 1]])

    max_mu = 0
    coefs = []
    mus = []

    for i in range(0, len(z)):
        mu = 0
        for j in range(0, len(data)):
            if data[j] - eps - (A + B * j) <= z[i][0] and data[j] + eps - (A + B * j) >= z[i][1]:
                mu += 1
        mus.append(mu)

        if mu > max_mu:
            max_mu = mu
            coefs = []
            coefs.append(i)

        if mu == max_mu:
            coefs.append(i)

    mode = []

    for i in range(0, len(z)):
        if i in coefs:
            mode.append(z[i])

    for k in range(0, len(mode) - 1):
        if mode[k][1] == mode[k + 1][0]:
            mode[k] = [mode[k][0], mode[k + 1][1]]

    return mode, max_mu, z, mus


result_0 = subinterval_frequencies_and_mode(A0, B0, eps, data)
result_1 = subinterval_frequencies_and_mode(A1, B1, eps, data)
mode_0 = result_0[0]
mode_1 = result_1[0]
max_mu_0 = result_0[1]
max_mu_1 = result_1[1]
z_0 = np.array(result_0[2])
z_1 = np.array(result_1[2])
mu_0 = result_0[3]
mu_1 = result_1[3]
plt.figure()
plt.plot(z_0[:, 0], mu_0, linewidth = 0.5, color = 'cyan')
plt.plot(z_1[:, 0], mu_1, linewidth = 0.5, color = 'sienna')
plt.xlabel('mV')
plt.ylabel('mu')
plt.savefig('mu.png')
plt.title('Частоты элементарных подинтервалов при вычислении моды модели')
plt.show()
print(f'mode_0 = {mode_0} mu_max_0 = {max_mu_0}')
print(f'mode_1 = {mode_1}, mu_max_1 = {max_mu_1}')


"""Коэфициент Жакара"""


def Jakar_coeff(A, B, eps, data):
    min_inc, max_inc = [], []
    min_inc.append(data[0] - eps - A)
    min_inc.append(data[0] + eps - A)
    max_inc.append(data[0] - eps - A)
    max_inc.append(data[0] + eps - A)
    for i in range(0, len(data)):
        min_inc[0] = max(min_inc[0], data[i] - eps - (A + B * i))
        min_inc[1] = min(min_inc[1], data[i] + eps - (A + B * i))
        max_inc[0] = min(max_inc[0], data[i] - eps - (A + B * i))
        max_inc[1] = max(max_inc[1], data[i] + eps - (A + B * i))
    JK = (min_inc[1] - min_inc[0]) / (max_inc[1] - max_inc[0])
    return JK

J0 = Jakar_coeff(A0, B0, eps, data)
J1 = Jakar_coeff(A1, B1, eps, data)
print(f'Ji_0 = {J0}, Ji_1 = {J1}')


"""Информационное множество задачи. Найдем вершины многоугольника"""


A , b = [], []
for i in range (0, len(data)):
    A.append([1, i])
    A.append([-1, -i])
    b.append(data[i] + eps)
    b.append(-data[i] + eps)
A = np.array(A)
b = np.array(b)
vertices = np.array(pyp.compute_polytope_vertices(A,b))

beta_0_min = min(vertices[:,0])
beta_0_max = max(vertices[:,0])
beta_1_min = min(vertices[:,1])
beta_1_max = max(vertices[:,1])

print(f'beta_0 in [{beta_0_min},{beta_0_max}]')
print(f'beta_1 in [{beta_1_min},{beta_1_max}]')


order = np.argsort(np.arctan2(vertices[:,1] - vertices[:,1].mean(), vertices[:,0] - vertices[:,0].mean()))

plt.figure()
plt.fill(vertices[:,0][order], vertices[:,1][order], edgecolor='tomato', color = 'brown', alpha = 0.3, linewidth = 0.5)
plt.scatter(vertices[:,0], vertices[:,1], color = 'brown', s = 0.5)
plt.plot([beta_0_min, beta_0_min], [beta_1_min, beta_1_max], linewidth = 0.5, color = 'cyan')
plt.plot([beta_0_max, beta_0_max], [beta_1_min, beta_1_max], linewidth = 0.5, color = 'cyan')
plt.plot([beta_0_min, beta_0_max], [beta_1_min, beta_1_min], linewidth = 0.5, color = 'cyan')
plt.plot([beta_0_min, beta_0_max], [beta_1_max, beta_1_max], linewidth = 0.5, color = 'cyan')
plt.xlabel('beta_0')
plt.ylabel('beta_1')
plt.title('Information set')
plt.savefig('inform_set.png')
plt.title('Информационное множество задачи.')
plt.show()


"""Коридор совместных зависимостей"""


max_val, min_val = [], []
for i in range (0, len(data)):
    minimum = 100
    maximum = 0
    for v in vertices:
        if v[0] + v[1] * i > maximum:
            maximum =  v[0] + v[1] * i
        if v[0] + v[1] * i < minimum:
            minimum = v[0] + v[1] * i
    min_val.append(minimum)
    max_val.append(maximum)

plt.figure()
plt.fill_between(data.index, np.array(data) + eps, np.array(data) - eps, color = 'lime', alpha = 0.3)
plt.plot(data.index, data, color = 'cyan', linewidth = 0.5)
plt.fill_between(np.arange(0, 200), np.array(min_val), np.array(max_val), color='sienna', alpha = 0.3)
plt.xlabel('n')
plt.ylabel('mV')
plt.title('Коридор совместных зависимостей')
plt.show()
plt.close()


"""Построение прогноза для n от -50 до 250"""


max_val_pr, min_val_pr = [], []
for i in range (-50, 250):
    minimum = 100
    maximum = 0
    for v in vertices:
        if v[0] + v[1] * i > maximum:
            maximum =  v[0] + v[1] * i
        if v[0] + v[1] * i < minimum:
            minimum = v[0] + v[1] * i
    min_val_pr.append(minimum)
    max_val_pr.append(maximum)

plt.figure()
plt.fill_between(data.index, np.array(data) + eps, np.array(data) - eps, color = 'lime', alpha = 0.3)
plt.fill_between(np.arange(-50, 250), np.array(min_val_pr), np.array(max_val_pr), color='sienna', alpha = 0.3)
plt.plot(data.index, data, color = 'cyan', linewidth = 0.5)
plt.xlabel('n')
plt.ylabel('mV')
plt.title('Построение прогноза для n от -50 до 250')
plt.show()
plt.close()








