#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from math import log as ln
from openpyxl import Workbook


def main(binary_path: str, composition_path: str, pressure: float, temperature: float, save_path: str):
    """
    Глвная расчётная функция, формирующая отчёт

    :param binary_path: String значение пути к .xlsx файлу бинарных значений
    :param composition_path: String значение пути к .xlsx файлу значений состава
    :param pressure: Float значение давления
    :param temperature: Float значение температуры
    :param save_path: String значение пути к папке для сохранения результата
    :return:
    """
    plt.style.use("seaborn-whitegrid")

    # Импорт параметров компонентов из экселя. Ипмортируется также состав смеси
    df = pd.read_excel(composition_path)
    df.drop(labels=['пар-ры'], axis=1, inplace=True)

    # Запись параметров компонентов смеси в словари
    POTG = {}  # parameters_of_the_gas2
    for col in df:
        POTG[col] = {'T_c': df[col][0], 'P_c': df[col][1],
                     'w_c': df[col][2], 'mole_c': df[col][3],
                     'Z_c': df[col][4], 'L_c': df[col][5], 'Psi_c': df[col][6]}

    # Определение количества компонентов
    N = df.shape[1]

    # импорт данных о бинарных коэффициентах взаимодействия
    c_ij = pd.read_excel(binary_path)
    c_ij.drop(labels=['c(ij)'], axis=1, inplace=True)
    c_ij.columns = [i for i in
                    range(N)]  # переименование названий колонн в соответствующие порядковые номера компонентов

    # Перезапись датафрейма из коэффициентов бинарного взаимодействия c_ij в матрицу
    cij = [[c_ij[col][i] for i in range(N)] for col in c_ij]

    # Ввод темобарических параметров. Создание названия эксель-файла для ввода резльтатов
    R = 0.00831
    T = temperature
    P = pressure
    excel_name = 'result' + '_' + str(P) + '_' + str(T) + '.xlsx'

    # Функции для расчета вириальных коэффициентов в ураавнении состояния
    def no_compl(array):
        ans = []
        for el in array:
            if el == el.real:
                ans.append(el.real)
        return ans

    def alpha(Omega):
        return Omega ** 3

    def betta(Z, Omega):
        return Z + Omega - 1

    def sigma(Z, Omega):
        return -Z + Omega * (0.5 + (Omega - 0.75) ** 0.5)

    def delta(Z, Omega):
        return -Z + Omega * (0.5 - (Omega - 0.75) ** 0.5)

    def afun(alpha, T_c, P_c, Psi):
        return (alpha * R ** 2 * T_c ** 2 * (1 + Psi * (1 - (T / T_c) ** 0.5)) ** 2) / P_c

    def bcdfun(becide, T_c, P_c):
        return becide * R * T_c / P_c

    # Расчет коэффициента распределения
    def K_i(w, T_c, P_c):
        P_Si = P_c * math.exp(5.373 * (1 + w) * (1 - (T_c / T)))
        return P_Si / P

    # Функция расчета функции F(V), определяющей фазовое состояние
    def F_V(z, K, eps):
        """
        z: list
        K: list
        eps: float
        """
        proiz = 0
        delen = 0
        for i in range(len(z)):
            proiz += z[i] * K[i]
            delen += z[i] / K[i]

        if proiz == 1:
            return 0
        elif delen == 1:
            return 1

        # !!!! Функции определены внутри другой функции. Так и должно быть!
        def function(x):
            res = 0
            for i in range(len(z)):
                res += z[i] * (K[i] - 1) / (x * (K[i] - 1) + 1)
            return res

        def bisection(a, c, b):
            if abs(function(c)) <= eps:
                return c
            if function(c) * function(b) > 0:
                res = bisection(a, (a + c) / 2, c)
                return res
            elif function(c) * function(a) > 0:
                res = bisection(c, (b + c) / 2, b)
                return res

        if proiz > 1 or delen > 1:
            if function(0) * function(1) < 0:
                return bisection(0, 0.5, 1)
            return 1 - eps

        elif proiz < 1:
            if len(otric) != 0:
                return max(otric)
        # fun for V < 0

        elif delen < 1:
            if len(polozhit) != 0:
                return min(polozhit)
        # fun for V > 1

    # Функции расчета мольных долей в фазах
    def y_i(z, K, V):
        """
        z: list
        K: list
        V: float
        """
        return [z[i] * K[i] / (V * (K[i] - 1) + 1) for i in range(N)]

    def x_i(z, K, V):
        """
        z: list
        K: list
        V: float
        """
        return [z[i] / (V * (K[i] - 1) + 1) for i in range(N)]

    # Функции расчета коэффициентов на основе параметров бинарного равновесия для кубического уравнения
    def a_mf(xy, aij):
        a_m = 0
        for i in range(N):
            for j in range(N):
                a_m += xy[i] * xy[j] * aij[i][j]
        return a_m


    def b_mf(xy, b):
        b_m = 0
        for i in range(N):
            for j in range(N):
                b_m += xy[i] * xy[j] * 0.5 * (b[i] + b[j])
        return b_m


    def cd_mf(xy, cd):
        cd_m = 0
        for i in range(N):
            cd_m += xy[i] * cd[i]
        return cd_m


    def A_mf(a_m):
        return a_m * P / (R * T) ** 2


    def B_mf(b_m):
        return b_m * P / (R * T)


    def CD_mf(cd_m):
        return cd_m * P / (R * T)


    def ln_f_VL(xy, z, B_m, A_m, C_m, D_m, a_m, c, d, c_m, d_m, B, C, D):
        koef_ln = {'ln(xyp)': [], 'ln(zBm)': [], 'A/CD': [], 'SUM': [], 'ln(zCmBm)': [], 'BizBm': [], 'zCD(i,m)': [],
                   'ln_f_VL': []}
        for i in range(N):
            summ = 0
            for j in range(N):
                summ += xy[j] * aij[i][j]
            koef_ln['ln(xyp)'].append(ln(xy[i] * P))
            koef_ln['ln(zBm)'].append(ln(z - B_m))
            koef_ln['A/CD'].append(A_m / (C_m - D_m))
            koef_ln['SUM'].append((2 * summ / a_m - (c[i] - d[i]) / (c_m - d_m)))
            koef_ln['ln(zCmBm)'].append(ln((z + C_m) / (D_m + z)))
            koef_ln['BizBm'].append(B[i] / (z - B_m))
            koef_ln['zCD(i,m)'].append((C[i] / (z + C_m) - D[i] / (z + D_m)))
            koef_ln['ln_f_VL'].append(
                koef_ln['ln(xyp)'][i] - koef_ln['ln(zBm)'][i] - koef_ln['A/CD'][i] * koef_ln['SUM'][i] *
                koef_ln['ln(zCmBm)'][i] + koef_ln['BizBm'][i] - koef_ln['A/CD'][i] * koef_ln['zCD(i,m)'][i])
        return [math.exp(koef_ln['ln_f_VL'][i]) for i in range(N)], koef_ln

    # Запись и округление в словари со свойствами газов рассчитанных по формулам вириальных коэффициеентов и КФР
    for gas in POTG:
        POTG[gas]['alpha'] = round(alpha(POTG[gas]['L_c']), 6)
        POTG[gas]['betta'] = round(betta(POTG[gas]['Z_c'], POTG[gas]['L_c']), 6)
        POTG[gas]['sigma'] = round(sigma(POTG[gas]['Z_c'], POTG[gas]['L_c']), 6)
        POTG[gas]['delta'] = round(delta(POTG[gas]['Z_c'], POTG[gas]['L_c']), 6)

        POTG[gas]['a'] = round(afun(POTG[gas]['alpha'], POTG[gas]['T_c'], POTG[gas]['P_c'], POTG[gas]['Psi_c']), 6)
        POTG[gas]['b'] = round(bcdfun(POTG[gas]['betta'], POTG[gas]['T_c'], POTG[gas]['P_c']), 6)
        POTG[gas]['c'] = round(bcdfun(POTG[gas]['sigma'], POTG[gas]['T_c'], POTG[gas]['P_c']), 6)
        POTG[gas]['d'] = round(bcdfun(POTG[gas]['delta'], POTG[gas]['T_c'], POTG[gas]['P_c']), 6)

        POTG[gas]['K'] = round(K_i(POTG[gas]['w_c'], POTG[gas]['T_c'], POTG[gas]['P_c']), 6)

    # Исходная мольная доля в смеси. Параметр не меняется
    z = [POTG[gas]['mole_c'] for gas in POTG]
    # Исходное значение КФР. Параметр будет пересчитываться неколько раз
    K = [POTG[gas]['K'] for gas in POTG]
    # proiz = 0
    # delen = 0
    eps = 0.00001

    V = F_V(z, K, eps)
    # print(V)

    # Запись результатов расчета фазовых составов в переменные
    y = y_i(z, K, V)
    x = x_i(z, K, V)

    a = [POTG[gas]['a'] for gas in POTG]
    b = [POTG[gas]['b'] for gas in POTG]
    c = [POTG[gas]['c'] for gas in POTG]
    d = [POTG[gas]['d'] for gas in POTG]

    aij = [[(1 - cij[i][j]) * (a[i] * a[j]) ** 0.5 for j in range(N)] for i in range(N)]

    Ay_m = A_mf(a_mf(y, aij))
    By_m = B_mf(b_mf(y, b))
    Cy_m = CD_mf(cd_mf(y, c))
    Dy_m = CD_mf(cd_mf(y, d))

    Ax_m = A_mf(a_mf(x, aij))
    Bx_m = B_mf(b_mf(x, b))
    Cx_m = CD_mf(cd_mf(x, c))
    Dx_m = CD_mf(cd_mf(x, d))

    # Полиномы для корректировки результатов расчета фугитивностей и КФР
    pol_K_y = [1, 0, 0, 0]
    pol_K_y[1] = Cy_m + Dy_m - By_m - 1
    pol_K_y[2] = Ay_m - Cy_m * By_m + Cy_m * Dy_m - By_m * Dy_m - Dy_m - Cy_m
    pol_K_y[3] = -(By_m * Cy_m * Dy_m + Cy_m * Dy_m + Ay_m * By_m)

    pol_K_x = [1, 0, 0, 0]
    pol_K_x[1] = Cx_m + Dx_m - Bx_m - 1
    pol_K_x[2] = Ax_m - Cx_m * Bx_m + Cx_m * Dx_m - Bx_m * Dx_m - Dx_m - Cx_m
    pol_K_x[3] = -(Bx_m * Cx_m * Dx_m + Cx_m * Dx_m + Ax_m * Bx_m)

    z_fact_y = np.roots(pol_K_y)
    z_fact_y = no_compl(z_fact_y)

    z_fact_x = np.roots(pol_K_x)
    z_fact_x = no_compl(z_fact_x)

    x_z_fact = round(min(z_fact_x), 6)
    y_z_fact = round(max(z_fact_y), 6)

    # print (pol_K_y, y_z_fact, sep = '\n')

    B_i = [B_mf(b[i]) for i in range(len(b))]

    C_i = [CD_mf(c[i]) for i in range(len(c))]

    D_i = [CD_mf(d[i]) for i in range(len(d))]

    # Расчет значений коэффициентов для расчета фугитивнойстей в жидкой и газообразной фазах
    a_m_x = a_mf(x, aij)
    a_m_y = a_mf(y, aij)

    c_m_x = cd_mf(x, c)
    c_m_y = cd_mf(y, c)

    d_m_x = cd_mf(x, d)
    d_m_y = cd_mf(y, d)

    # Расчет фугитивностей
    f_V_y, koef_ln_y = ln_f_VL(y, y_z_fact, By_m, Ay_m, Cy_m, Dy_m, a_m_y, c, d, c_m_y, d_m_y, B_i, C_i, D_i)

    f_L_x, koef_ln_x = ln_f_VL(x, x_z_fact, Bx_m, Ax_m, Cx_m, Dx_m, a_m_x, c, d, c_m_x, d_m_x, B_i, C_i, D_i)

    # Определение погрешности вычислений КФР относительно предыдущего шага
    otnosh = [abs(f_L_x[i] / f_V_y[i] - 1) for i in range(len(f_L_x))]

    # Корректировка вычислений до получения погрешности, не превышающей 0,0001
    K_korr = [el for el in K]
    flag = 0
    while max(otnosh) > 0.0001:
        flag += 1
        for i in range(len(K_korr)):
            K_korr[i] *= f_L_x[i] / f_V_y[i]

        y = y_i(z, K_korr, V)
        x = x_i(z, K_korr, V)

        Ay_m = A_mf(a_mf(y, aij))
        By_m = B_mf(b_mf(y, b))
        Cy_m = CD_mf(cd_mf(y, c))
        Dy_m = CD_mf(cd_mf(y, d))

        Ax_m = A_mf(a_mf(x, aij))
        Bx_m = B_mf(b_mf(x, b))
        Cx_m = CD_mf(cd_mf(x, c))
        Dx_m = CD_mf(cd_mf(x, d))

        pol_K_y = [1, 0, 0, 0]
        pol_K_y[1] = Cy_m + Dy_m - By_m - 1
        pol_K_y[2] = Ay_m - Cy_m * By_m + Cy_m * Dy_m - By_m * Dy_m - Dy_m - Cy_m
        pol_K_y[3] = -(By_m * Cy_m * Dy_m + Cy_m * Dy_m + Ay_m * By_m)

        pol_K_x = [1, 0, 0, 0]
        pol_K_x[1] = Cx_m + Dx_m - Bx_m - 1
        pol_K_x[2] = Ax_m - Cx_m * Bx_m + Cx_m * Dx_m - Bx_m * Dx_m - Dx_m - Cx_m
        pol_K_x[3] = -(Bx_m * Cx_m * Dx_m + Cx_m * Dx_m + Ax_m * Bx_m)

        z_fact_y = no_compl(np.roots(pol_K_y))

        z_fact_x = no_compl(np.roots(pol_K_x))

        x_z_fact = round(min(z_fact_x), 6)
        y_z_fact = round(max(z_fact_y), 6)

        B_i = [B_mf(b[i]) for i in range(len(b))]

        C_i = [CD_mf(c[i]) for i in range(len(c))]

        D_i = [CD_mf(d[i]) for i in range(len(d))]

        a_m_x = a_mf(x, aij)
        a_m_y = a_mf(y, aij)

        c_m_x = cd_mf(x, c)
        c_m_y = cd_mf(y, c)

        d_m_x = cd_mf(x, d)
        d_m_y = cd_mf(y, d)

        f_V_y, koef_ln_y = ln_f_VL(y, y_z_fact, By_m, Ay_m, Cy_m, Dy_m, a_m_y, c, d, c_m_y, d_m_y, B_i, C_i, D_i)

        f_L_x, koef_ln_x = ln_f_VL(x, x_z_fact, Bx_m, Ax_m, Cx_m, Dx_m, a_m_x, c, d, c_m_x, d_m_x, B_i, C_i, D_i)

        otnosh = [abs(f_L_x[i] / f_V_y[i] - 1) for i in range(len(f_L_x))]

    # Определяем наличие и количество жидкой фазы по результатам расчета
    L = 1 - V

    wb = Workbook()
    Ws = wb.active

    Ws.append((L, V))

    if L * 100 > 0.05:
        message = 'Да, есть жидкая фаза'
        Ws.append(['Да, есть жидкая фаза'])
    else:
        message = 'Нет жидкой фазы'
        Ws.append(['Нет жидкой фазы'])

    # Считаем итогово
    y = y_i(z, K, V)
    x = x_i(z, K, V)

    Ws.append([round(el, 10) for el in x])

    Ws.append([round(el, 10) for el in y])

    # СОХРАНЯЕМ РЕЗУЛЬТАТ РАСЧЕТА В ФАЙЛ ЭКСЕЛЬ
    wb.save(f'{save_path}./{excel_name}')
    return message
