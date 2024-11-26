import main as rd
import math



def get_k(sect_count: int, delta_X_init: float, l: float, key: str) -> float: # key 0 - subsonic, 1 - supersonic

    # решается итерационым методом Ньютона

    start_delta_X = delta_X_init

    k = 0.99  # начальное приближение для k

    while (True):

        fk = 0.
        fkp = 0.
        for n in range(1, sect_count+1):
            fk += k**n
            fkp += n*k**(n-1)

        fk = fk - l/delta_X_init

        k1 = k - fk/fkp
        if (abs((k1 - k)/k1) < 0.00001 ):
            if (k <= 1 and key == "Subsonic"):
                print("1")
                break
            elif (k > 1 and key == "Supersonic"):
                print("2")
                break
            else:
                print("Введите большее количество сечений")
                k = -1.
                break
        else: k = k1

    return k


def init_mesh(x_list: list[float], d_list: list[float], n_section: int, gemtr_list: list[float]) -> [list[float], list[float], int]:

    l_kam = gemtr_list[0]
    d_kam = gemtr_list[1]
    x_kr = gemtr_list[2]
    d_kr = gemtr_list[3]

    l_supsn = x_list[len(x_list)-1] - x_kr
    l_subsn = x_kr - l_kam

    if (l_supsn/l_subsn <= 2):

        chbr_sect_count = math.floor(0.1*n_section + 0.5)
        subsn_sect_count = math.floor(0.4 * n_section + 0.5)
        supsn_sect_count = math.floor(0.5 * n_section + 0.5)

    else:

        chbr_sect_count = math.floor(0.1 * n_section)
        subsn_sect_count = math.floor(0.3 * n_section)
        supsn_sect_count = math.floor(0.6 * n_section)

    number_section = chbr_sect_count + subsn_sect_count + supsn_sect_count



    # Определяем шаги по секциям

    delta_X_chmbr = l_kam/chbr_sect_count

    # Определение коэффициента уменьшения k для дозвуковой части сопла

    k_subsn = get_k(subsn_sect_count, delta_X_chmbr, l_subsn, key = "Subsonic")

    control_l_subsn = 0                         # проверяем равенство суммы длин всех секций дозвука и длины дозвука
    k = k_subsn
    for i in range(1, subsn_sect_count+1):
        control_l_subsn += k * delta_X_chmbr
        k *= k_subsn

    correction_first_dx = control_l_subsn - l_subsn
    first_dx_sub = delta_X_chmbr*k_subsn - correction_first_dx

    # Определение коэффициента уменьшения k для сверхзвуковой части сопла

    k_supsn = get_k(supsn_sect_count, delta_X_chmbr*(k_subsn**subsn_sect_count), l_supsn, key="Supersonic")

    control_l_supsn = 0  # проверяем равенство суммы длин всех секций сверхзвука и длины сверхзвука
    k = k_supsn
    for i in range(1, supsn_sect_count + 1):
        control_l_supsn += k * (delta_X_chmbr*k_subsn**subsn_sect_count)
        k *= k_supsn

    correction_last_dx = control_l_supsn - l_supsn

    # заполнение массива X

    sections_x_list = [0] * (number_section + 1)
    sections_d_list = [0] * (number_section + 1)

    sections_x_list[0] = x_list[0]                      # инициализация камеры сгорания
    for i in range(1,chbr_sect_count + 1):
        sections_x_list[i] = sections_x_list[i-1] + delta_X_chmbr

    sections_x_list[chbr_sect_count+1] = sections_x_list[chbr_sect_count] + first_dx_sub    # инициализация дозвуковой части
    k = k_subsn**2
    for i in range(chbr_sect_count+2, chbr_sect_count + subsn_sect_count+1):
        sections_x_list[i] = sections_x_list[i - 1] + delta_X_chmbr*k
        k *= k_subsn

    start_dx = k_supsn*delta_X_chmbr*k_subsn**(subsn_sect_count)

    sections_x_list[chbr_sect_count + subsn_sect_count + 1] = sections_x_list[chbr_sect_count + subsn_sect_count] + start_dx   # инициализация сверхзвуковой части
    k = k_supsn
    for i in range(chbr_sect_count + subsn_sect_count + 2, number_section +1):
        sections_x_list[i] = sections_x_list[i - 1] + start_dx * k
        k *= k_supsn

    sections_x_list[chbr_sect_count + subsn_sect_count + supsn_sect_count] = sections_x_list[chbr_sect_count + subsn_sect_count + supsn_sect_count] - correction_last_dx

    sections_d_list[0] = d_list[0]
    for i in range(1,number_section+1):
        for j in range(1, len(x_list)-1):
            if ((x_list[j-1] <= sections_x_list[i]) and (sections_x_list[i] < x_list[j])):
                sections_d_list[i] = d_list[j-1] + (sections_x_list[i] - x_list[j-1])*(d_list[j] - d_list[j-1])/(x_list[j] - x_list[j-1])
                break;

    sections_d_list[-1] = d_list[-1]

    print(sections_d_list[-1])

    return sections_x_list, sections_d_list, [number_section, chbr_sect_count, subsn_sect_count, supsn_sect_count]


def set_geom_data_table(x_section: list[float], d_section: list[float],
                        number_section: int, gemtr_list: list[float]) -> list[list[float]]:

    l_kam = gemtr_list[0]
    d_kam = gemtr_list[1]
    x_kr = gemtr_list[2]
    d_kr = gemtr_list[3]

    F_kr = math.pi*(d_kr*10**-3)**2/4

    # таблица данных 1

    D_otn = [0]*(number_section+1)
    F_sect = [0]*(number_section+1)
    F_otn = [0]*(number_section+1)
    dx = [0]*(number_section)
    dxs = [0] * (number_section)
    dS = [0] * (number_section)

    for i in range(number_section+1):

        D_otn[i] = d_section[i]/d_kr
        F_sect[i] = math.pi*(d_section[i]*10**-3)**2/4
        F_otn[i] = F_sect[i]/F_kr


    for i in range(number_section):

        dx[i] = (x_section[i + 1] - x_section[i]) * 10 ** -3
        dxs[i] = math.sqrt((d_section[i + 1] - d_section[i]) ** 2 + (x_section[i + 1] - x_section[i]) ** 2) * 10 ** -3
        dS[i] = 0.5 * math.pi * (d_section[i + 1] + d_section[i]) * 10 ** -3 * dxs[i]

    table = [D_otn, F_sect, F_otn, dx, dxs, dS]

    return table

def set_cooling_path_table(x_section, d_section, section: list[list[int]], cooling_path_param: list[float]) -> list[list[float]]:

    # параметры тракта охлаждения

    inner_wall = cooling_path_param[0]*10**-3
    outer_wall = cooling_path_param[1]*10**-3
    delt_r = cooling_path_param[2]*10**-3
    betta = cooling_path_param[3]
    height = cooling_path_param[4]*10**-3


    # количество секций по зонам

    number_section = section[0]
    chbr_sec_count = section[1]
    subsn_sec_count = section[2]


    # минимальный и максимальный шаг оребрения
    tN_min = delt_r + 1.5 *10**-3
    tN_max = delt_r + 6 *10**-3

    # количество ребер в критическом сечении сопла

    D_sr = d_section[chbr_sec_count+subsn_sec_count]*10**-3 * (1 + (2*inner_wall + height) / d_section[chbr_sec_count+subsn_sec_count]/10**-3)
    Nr_kr = math.floor((math.pi*D_sr*math.cos(betta/180*math.pi))/tN_min)

    Nr = [0]*number_section
    t = [0]*number_section
    tN = [0]*number_section
    f = [0]*number_section
    D_g = [0]*number_section

    # заполнение таблицы данных о проточной части для камеры и дозвуковой части

    kn = 1
    Dkn = (tN_max * kn * Nr_kr)/(math.pi * math.cos(betta/180*math.pi)) - (2*inner_wall + height)
    for i in range(chbr_sec_count+subsn_sec_count, 0, -1):
        if (d_section[i-1]*10**-3 < Dkn):
            Nr[i-1] = Nr_kr * kn
            D_sr = d_section[i-1] * 10 ** -3 * (1 + (2 * inner_wall + height) / d_section[i-1]/10**-3)
            t[i-1] = math.pi * D_sr / Nr[i-1]
            tN[i-1] = t[i-1] * math.cos(betta/180*math.pi)
            f[i-1] = tN[i-1] * height * (1 - delt_r/tN[i-1]) * Nr[i-1]
            D_g[i-1] = 2 * height * (tN[i-1] - delt_r) / (tN[i-1] - delt_r + height)
        else :
            kn *= 2
            Dkn = (tN_max * kn * Nr_kr) / (math.pi * math.cos(betta / 180 * math.pi)) - (2 * inner_wall + height)
            Nr[i - 1] = Nr_kr * kn
            D_sr = d_section[i - 1] * 10 ** -3 * (1 + (2 * inner_wall + height) / d_section[i - 1] / 10 ** -3)
            t[i - 1] = math.pi * D_sr / Nr[i - 1]
            tN[i - 1] = t[i - 1] * math.cos(betta / 180 * math.pi)
            f[i - 1] = tN[i - 1] * height * (1 - delt_r / tN[i - 1]) * Nr[i - 1]
            D_g[i - 1] = 2 * height * (tN[i - 1] - delt_r) / (tN[i - 1] - delt_r + height)


# заполнение таблицы данных о проточной части для сверхзвуковой части сопла

    kn = 1
    Dkn = (tN_max * kn * Nr_kr)/(math.pi * math.cos(betta/180*math.pi)) - (2*inner_wall + height)
    for i in range(chbr_sec_count + subsn_sec_count + 1, number_section + 1):
        if (d_section[i-1]*10**-3 < Dkn):
            Nr[i-1] = Nr_kr * kn
            D_sr = d_section[i-1] * 10 ** -3 * (1 + (2 * inner_wall + height) / d_section[i-1]/10**-3)
            t[i-1] = math.pi * D_sr / Nr[i-1]
            tN[i-1] = t[i-1] * math.cos(betta/180*math.pi)
            f[i-1] = tN[i-1] * height * (1 - delt_r/tN[i-1]) * Nr[i-1]
            D_g[i-1] = 2 * height * (tN[i-1] - delt_r) / (tN[i-1] - delt_r + height)
        else :
            kn *= 2
            Dkn = (tN_max * kn * Nr_kr) / (math.pi * math.cos(betta / 180 * math.pi)) - (2 * inner_wall + height)
            Nr[i - 1] = Nr_kr * kn
            D_sr = d_section[i - 1] * 10 ** -3 * (1 + (2 * inner_wall + height) / d_section[i - 1] / 10 ** -3)
            t[i - 1] = math.pi * D_sr / Nr[i - 1]
            tN[i - 1] = t[i - 1] * math.cos(betta / 180 * math.pi)
            f[i - 1] = tN[i - 1] * height * (1 - delt_r / tN[i - 1]) * Nr[i - 1]
            D_g[i - 1] = 2 * height * (tN[i - 1] - delt_r) / (tN[i - 1] - delt_r + height)


    table = [Nr, t, tN, f, D_g]

    return table


def get_lambda(q: float, k: float, key: str) -> float:

    if (key == "Subsonic"):
        lmbd  = 0.3
    elif (key == "Supersonic"):
        lmbd  = 1.01
    while (True):
        if (key == "Subsonic"):
            f = lmbd * (1 - (k - 1) / (k + 1) * lmbd ** 2) ** (1 / (k - 1)) * ((k + 1) / 2) ** (1 / (k - 1)) - q
            delta_lamda = 0.01
            fd = ((lmbd + delta_lamda) * (1 - (k - 1) / (k + 1) * (lmbd + delta_lamda) ** 2) ** (1 / (k - 1)) * (
                        (k + 1) / 2) ** (1 / (k - 1))) / delta_lamda
        elif (key == "Supersonic"):
            f = lmbd * (1 - (k - 1) / (k + 1) * lmbd ** 2) ** (1 / (k - 1)) * ((k + 1) / 2) ** (1 / (k - 1)) - q
            delta_lamda = 0.01
            fd = ((lmbd - delta_lamda) * (1 - (k - 1) / (k + 1) * (lmbd - delta_lamda) ** 2) ** (1 / (k - 1)) * (
                        (k + 1) / 2) ** (1 / (k - 1))) / -delta_lamda


        lambda1 = lmbd - f/fd

        if (abs((lambda1 - lmbd) / lambda1) < 0.000001):
                lambda1
                break
        else:
            lmbd = lambda1


    return lambda1

def calc_heat_flow(x_section: list[float], table_geom: list[list[float]], section: list[list[float]], gemtr_list: list[float],
                      thermophysical_parameters: list[float]) -> list[list[float]]:

    l_kam = gemtr_list[0]
    d_kam = gemtr_list[1]
    x_kr = gemtr_list[2]
    d_kr = gemtr_list[3]
    x_cooling_change = gemtr_list[4]

    number_section = section[0]
    chbr_sec_count = section[1]
    subsn_sec_count = section[2]

    D_otn = table_geom[0]
    F_sect = table_geom[1]
    F_otn = table_geom[2]
    dx = table_geom[3]
    dxs = table_geom[4]
    dS = table_geom[5]

    Pr = thermophysical_parameters[0]
    mu_T0 = thermophysical_parameters[1]
    R_T0 = thermophysical_parameters[2]
    T_st_usl = thermophysical_parameters[3]
    T_0g = thermophysical_parameters[4]
    cp_st_usl = thermophysical_parameters[5]
    cp_T0 = thermophysical_parameters[6]
    k = thermophysical_parameters[7]
    p_k = thermophysical_parameters[8]
    epsilon_g = thermophysical_parameters[9]
    epsilon_st = thermophysical_parameters[10]
    fi = thermophysical_parameters[11]
    T_k = thermophysical_parameters[12]

    lmbd = [0] * (number_section + 1)
    betta = [0] * (number_section + 1)
    S = [0] * (number_section + 1)
    T_st_otn = [0] * (number_section + 1)
    q_convect = [0] * (number_section + 1)
    q_rad = [0] * (number_section + 1)
    q_sum = [0] * (number_section + 1)

    if F_otn[0] >= 3.5:
        epsilon = 1
    else:
        epsilon = float(input("Относительная площадь камеры сгорания меньше 3,5, введите epsilon"))

    # для лучитсых тепловых потоков

    epsilon_st_eff = (epsilon_st + 1) / 2

    c0 = 5.67

    q_rad_Km = epsilon_st_eff * epsilon_g * c0 * (T_k/100)**4

    q_l_ks = fi * q_rad_Km

    for i in range(number_section + 1):

        if (i <= chbr_sec_count + subsn_sec_count):
            q = 1 / (D_otn[i]) ** 2
            lmbd[i] = get_lambda(q, k, key="Subsonic")
        else:
            q = 1 / (D_otn[i]) ** 2
            lmbd[i] = get_lambda(q, k, key="Supersonic")

            # lambda_array.append(lmbd)
            # q_array.append(1 / D_otn[i] ** 2)

        # расчет конвективного теплового потока

        cp_sr = (cp_st_usl + cp_T0) / 2
        T_st_otn[i] = T_st_usl / T_0g

        S[i] = (2.065 * cp_sr * (T_0g - T_st_usl) * mu_T0 ** 0.15) / (
                    (R_T0 * T_0g) ** 0.425 * (1 + T_st_otn[i]) ** 0.595 * (3 + T_st_otn[i]) ** 0.15)

        alpha = 1.813 * (2 / (k + 1)) ** (0.85 / (k - 1)) * ((2 * k) / (k + 1)) ** 0.425
        betta[i] = lmbd[i] * ((k - 1) / (k + 1)) ** 0.5

        Z = (1.769 * (1 - betta[i] ** 2 + betta[i] ** 2 * (
                    1 - 0.086 * (1 - betta[i] ** 2) / (1 - T_st_otn[i] - 0.1 * betta[i] ** 2))) / (
                         1 - 0.086 * (1 - betta[i] ** 2) / (1 - T_st_otn[i] - 0.1 * betta[i] ** 2))) ** 0.54

        A = 0.01352
        B = 0.4842 * alpha * A * Z ** 0.075

        q_convect[i] = B * ((1 - betta[i] ** 2) * epsilon * p_k ** 0.85 * S[i]) / (D_otn[i]**1.82 * (d_kr * 10**-3)**0.15 * Pr**0.58)

        # расчет лучистого теплового потока

        if (x_section[i] < 50):
            q_rad[i] = 0.25 * q_l_ks
        elif (D_otn[i] >= 1.2 and i < chbr_sec_count + subsn_sec_count):
            q_rad[i] = q_l_ks
        elif (D_otn[i] < 1.2 and i <= chbr_sec_count + subsn_sec_count):
            q_rad[i] = q_l_ks * (1 - 12.5 * (1.2 - D_otn[i])**2)
        else:
            q_rad[i] = 0.5 * q_l_ks / D_otn[i]**2

        q_sum[i] = q_convect[i] + q_rad[i]

    return [lmbd, betta, S, T_st_otn, q_convect, q_rad, q_sum]


def calc_cooling_path(x_section: list[float], table_geom, section, gemtr_list, cooler_data, heat_flow_table):

    l_kam = gemtr_list[0]
    d_kam = gemtr_list[1]
    x_kr = gemtr_list[2]
    d_kr = gemtr_list[3]
    x_cooling_change = gemtr_list[4]

    D_otn = table_geom[0]
    F_sect = table_geom[1]
    F_otn = table_geom[2]
    dx = table_geom[3]
    dxs = table_geom[4]
    dS = table_geom[5]

    number_section = section[0]
    chbr_sec_count = section[1]
    subsn_sec_count = section[2]

    q_sum = heat_flow_table[6]

    ro_cool = cooler_data[0]
    T_cool = cooler_data[1]
    C_cool = cooler_data[2]
    m_cooler = cooler_data[3]
    T_start_cooler = cooler_data[4]


    # Определяем точку смены функциональных схем охлаждения
    for i in range(number_section):
        if i == 90:
            a = 0
        if (x_section[i] < x_cooling_change) and (x_cooling_change <= x_section[i+1]):
            change_section = i
        if (x_cooling_change > x_section[number_section]):
            change_section = number_section - 1

    T_cooler = [0] * (change_section+2)
    T_cooler[change_section+1] = T_start_cooler

    for i in range(change_section,-1,-1):

        dT = 30

        while(True):

            T_sr = T_start_cooler + 0.5 * dT

            for j in range(len(T_cool)-1):
                if (T_cool[j] > T_sr) and (T_sr <= T_cool[j+1]):
                    C_sr = C_cool[j] + (T_sr - T_cool[j]) * (C_cool[j+1] - C_cool[j]) / (T_cool[j+1] - T_cool[j])
                    break
                if (T_sr > T_cool[len(T_cool)-1]):
                    C_sr = C_cool[len(T_cool)-2] + (T_sr - T_cool[len(T_cool)-2]) * (C_cool[len(T_cool)-1] - C_cool[len(T_cool)-2]) / (
                                T_cool[len(T_cool)-1] - T_cool[len(T_cool)-2])
                    break

            dT = 0.5 * (q_sum[i] + q_sum[i+1]) * dS[i] / (m_cooler * C_sr)

            T_sr = T_start_cooler + 0.5 * dT

            C_sr1 = C_sr
            for j in range(len(T_cool)-1):
                if (T_cool[j] > T_sr) and (T_sr <= T_cool[j+1]):
                    C_sr = C_cool[j] + (T_sr - T_cool[j]) * (C_cool[j+1] - C_cool[j]) / (T_cool[j+1] - T_cool[j])
                    break
                if (T_sr > T_cool[len(T_cool)-1]):
                    C_sr = C_cool[len(T_cool)-2] + (T_sr - T_cool[len(T_cool)-2]) * (C_cool[len(T_cool)-1] - C_cool[len(T_cool)-2]) / (
                                T_cool[len(T_cool)-1] - T_cool[len(T_cool)-2])
                    break

            if (abs(C_sr - C_sr1)/C_sr < 0.05):
                T_start_cooler = T_start_cooler + dT
                break

        T_cooler[i] = T_start_cooler




def heat_protection_main(X_list: list[float], D_list: list[float], n_section: int, gemtr_list: list[float],
                         cooling_path_param: list[float], thermophysical_parameters: list[float], cooler_data: list[float]) -> None:

    x_section, d_section, section = init_mesh(X_list, D_list, n_section, gemtr_list)  # [number_section, chbr_sect_count, subsn_sect_count, supsn_sect_count]

    table_geom = set_geom_data_table(x_section, d_section, section[0], gemtr_list)  # [D_otn, F_sect, F_otn, dx, dxs, dS]

    cooling_path_table = set_cooling_path_table(x_section, d_section, section, cooling_path_param)

    heat_flow_table = calc_heat_flow(x_section, table_geom, section, gemtr_list, thermophysical_parameters)

    calc_cooling_path(x_section, table_geom, section, gemtr_list, cooler_data, heat_flow_table)

    print(x_section[-1])








