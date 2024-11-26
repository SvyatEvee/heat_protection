import pandas as pd
import heat_protection as htprt

def get_init_data() -> [list[float]]:

    df = pd.read_excel("data.xlsx", header = 0)
    x_data = list(df["x"])
    d_data = list(df["D"])

    x_data.append(df["Xkr"][0])
    d_data.append(df["Dkr"][0])

    x_data.append(df["xkam"][0])
    d_data.append(df["Dkam"][0])

    number_of_section = 100

    return x_data, d_data, number_of_section


def get_cooler_data() -> list[float]:

    file = pd.read_excel("c_cooler.xlsx", header = 0)
    ro_cool = list(file["ro"])
    T_cool = list(file["T"])
    C_cool = list(file["C"])
    m_cooler = 6.571
    T_start_cooler = 293

    return [T_cool, C_cool, ro_cool, m_cooler, T_start_cooler]

if __name__ == "__main__":

    X_list, D_list, n_section = get_init_data()


    # Параметры сопла
    gemtr_list = [] # [l_kam, D_kam, X_kr, D_kr, x_cooling_change] # длина камеры должна быть больше 50 мм

    x_cooling_change = 1500 # точка смены проточного охлаждения на радиационное

    gemtr_list.append(X_list.pop())
    gemtr_list.append(D_list.pop())
    gemtr_list.append(X_list.pop())
    gemtr_list.append(D_list.pop())
    gemtr_list.append(x_cooling_change) # мм


    # Параметры тракта охлаждения
    cooling_path_param = [] # [inner_wall, outer_wall, delt_r, betta, height]

    inner_wall = 1
    outer_wall = 3
    delt_r = 1
    betta = 0
    height = 6

    cooling_path_param.append(inner_wall)
    cooling_path_param.append(outer_wall)
    cooling_path_param.append(delt_r)
    cooling_path_param.append(betta)
    cooling_path_param.append(height)

    # Теплофизические параметры продуктов сгорания

    thermophysical_parameters = [] # [Pr, mu_T0, R_T0, T_st_usl, T_0g, cp_st_usl, cp_T0, k, p_k, epsilon_g, epsilon_st]

    Pr = 0.75
    mu_T0 = 5.98 * 10**-5
    R_T0 = 509
    T_st_usl = 1000
    T_0g = 2000
    cp_st_usl = 2029
    cp_T0 = 2235
    k = 1.29
    p_k = 10 * 10**6
    epsilon_g = 0.25
    epsilon_st = 0.8
    fi = 0.9
    T_k = 3750


    thermophysical_parameters.append(Pr)
    thermophysical_parameters.append(mu_T0)
    thermophysical_parameters.append(R_T0)
    thermophysical_parameters.append(T_st_usl)
    thermophysical_parameters.append(T_0g)
    thermophysical_parameters.append(cp_st_usl)
    thermophysical_parameters.append(cp_T0)
    thermophysical_parameters.append(k)
    thermophysical_parameters.append(p_k)
    thermophysical_parameters.append(epsilon_g)
    thermophysical_parameters.append(epsilon_st)
    thermophysical_parameters.append(fi)
    thermophysical_parameters.append(T_k)


    cooler_data = get_cooler_data()

    htprt.heat_protection_main(X_list, D_list, n_section, gemtr_list, cooling_path_param, thermophysical_parameters, cooler_data)