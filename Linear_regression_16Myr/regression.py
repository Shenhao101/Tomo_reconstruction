# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:27:57 2024

linear regression

This code aimed to find the linear relationship between pCO2, tectonic degassing and weathering proxy

@author: shenhao
@email: shenhao@mail.iggcas.ac.cn
"""
import pandas as pd
import numpy as np
import csv
import scipy
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn import linear_model
import matplotlib.pyplot as plt



def load_carbon_flux(file):
    df = pd.read_csv(file)
    age = np.array(df['# time'][0:65])
    carbon_flux = np.array(df['total_subducted_mean  (Mt C/yr)'][0:65])

    # Normalization
    flux_max = carbon_flux.max()
    flux_min = carbon_flux.min()
    carbon_flux_normalized = (carbon_flux-flux_min) / (flux_max-flux_min)
    
    return age, carbon_flux_normalized



def load_Li_isotope(file):
    df = pd.read_excel(file, sheet_name='All Foram')
    Age = np.array(df['Age'])[3:]
    Li_isotope = np.array(df['Li'])[3:]
        
    
    # 5 points running mean
    num = 0 
    Age_new = []
    Li_isotope_new = []
    Age_temp = 0
    Li_isotope_temp = 0
    for i in range(len(Age)):
        num += 1
        Age_temp += Age[i]
        Li_isotope_temp += Li_isotope[i]
        if num == 5:
            Age_new.append(Age_temp/num)
            Li_isotope_new.append(Li_isotope_temp/num)
            num = 0 
            Age_temp = 0
            Li_isotope_temp = 0
    if num != 0:
        Age_new.append(Age_temp/num)
        Li_isotope_new.append(Li_isotope_temp/num)
    
    
    # interpolate to 1~65 with interval of 1 Ma
    Age_interp = np.arange(1,66)
    f = interpolate.interp1d(Age_new, Li_isotope_new)
    Li_interp = f(Age_interp)
    
# =============================================================================
#     # standardization
#     mean = np.mean(Li_interp)
#     std = np.std(Li_interp) # Standard Deviation
#     Li_standardization = (Li_interp - mean) / std
# =============================================================================
    
    # nomalization
    Li_max = Li_interp.max()
    Li_min = Li_interp.min()
    Li_normalization = (Li_interp-Li_min) / (Li_max-Li_min)
    
    return Age_interp, Li_normalization



def load_CO2(file):
    f = open(file, 'r', encoding=u'utf-8', errors='ignore')
    reader = csv.reader(f)
    reader = list(reader)
    age = []
    CO2 = []
    for i in range(len(reader)):
        age.append(float(reader[i][0]))
        CO2.append(float(reader[i][1]))
    age = np.array(age)
    CO2 = np.array(CO2)
    
    # interpolate to 1~65 with interval of 1 Ma
    age_interp = np.arange(1,66)
    f = interpolate.interp1d(age, CO2)
    CO2_interp = f(age_interp)
    
    
    # normalization
    CO2_max = CO2_interp.max()
    CO2_min = CO2_interp.min()
    CO2_normalization = (CO2_interp-CO2_min) / (CO2_max-CO2_min)
    

    # calculate feedback strength (Caves et al., 2016)
    CO2_modern = 278 # pre-industrial CO2 (Foster et al., 2015)
    # feedback strength relative to modern 
    R_CO2 = CO2_interp / CO2_modern
    R_fs = np.log2(R_CO2) + 1


    return age_interp, CO2_normalization, R_fs



def load_temperature(file):
    f = open(file, 'r')
    age = []
    temperature = []
    for each_line in f.readlines():
        each_line = each_line.strip('\n')
        each_line = each_line.split()
        age.append(float(each_line[0]))
        temperature.append(float(each_line[1]))
    age = np.array(age)
    temperature = np.array(temperature)
    
    age_interp = np.arange(1,66)
    f = interpolate.interp1d(age, temperature, kind='cubic')
    temp_interp = f(age_interp)
    
    # normalization
    temp_max = temp_interp.max()
    temp_min = temp_interp.min()
    temp_normalization = (temp_interp-temp_min) / (temp_max-temp_min)
    return age_interp, temp_normalization



def cosine_taper(freqs, flimit):
    fl1, fl2 = flimit
    taper = np.zeros_like(freqs)
    
    a = (fl1 <= freqs) & (freqs <= fl2)
    taper[a] = 0.5 * (1.0 + np.cos(np.pi * (freqs[a] - fl1) / (fl2 - fl1)))

    b = freqs < fl1
    taper[b] = 1.0

    return taper
    
    

def slide_correlation(Age_CO2, CO2, pCO2_pred, window, plot):
    corr_coef_slide = []
    p_value_slide = []
    for i in range(len(CO2)-window):    
        corr_coef, p_value = pearsonr(CO2[i:i+window+1], pCO2_pred[i:i+window+1])
        corr_coef_slide.append(corr_coef)
        p_value_slide.append(p_value)
    corr_coef_slide = np.asarray(corr_coef_slide)
    p_value_slide = np.asarray(p_value_slide)    
    
    Age_slide = Age_CO2[0:len(CO2)-window] + window/2
    
    # plot slide-window correlation coefficient with p<0.05
    plot_index = np.where(p_value_slide<0.05)
    if len(plot_index) !=0 and plot == True:
        Age_plot = Age_slide[plot_index]
        corr_coef_plot = corr_coef_slide[plot_index]
        
    fig, ax = plt.subplots(1)
    ax.errorbar(Age_plot, corr_coef_plot, xerr=window/2, fmt='.--', color='#4489C8', linewidth=1.5,
                ecolor='#008F91', elinewidth=1, capsize=2, capthick=1)
    #ax.plot(Age_slide, corr_coef_slide, marker='o')
    ax.set_xlim([0, 66])
    #ax.set_ylim([0.4, 1])
    
    return corr_coef_slide, p_value_slide, Age_slide
    
    
    
if __name__ == '__main__':
    
    # load mean normalized carbon flux
    file = 'subducted_carbon.csv'
    Age_flux, carbon_flux_mean = load_carbon_flux(file)
    
# =============================================================================
#     # load random signal
#     file = 'random_signal.csv'
#     df = pd.read_csv(file)
#     Age_noise = np.array(df['# time'])
#     signal_noise = np.array(df['random_signal'])
# =============================================================================
    

    # load Li isotope
    file = 'Li_isotope.xlsx'
    Age_Li, Li_isotope = load_Li_isotope(file)

    
    # load CO2
    file = 'CO2_CenCO2PIP_2023.csv'
    Age_CO2, CO2, R_fs = load_CO2(file)
# =============================================================================
#     # load temperature 
#     file = 'temperature_smooth_Hansen2013.txt' 
#     Age_CO2, CO2 = load_temperature(file)
# =============================================================================
    
    
    # # linear regression (the result is the same with curve fitting method)
    # x = []
    # for i in range(len(Age_flux)):
        
    # #x.append([carbon_flux_mean[i], carbon_flux_mean[i]*Sr_ratio[i],
    # #         carbon_flux_mean[i]*Li_isotope[i], carbon_flux_mean[i]*Or_isotope[i]])
    # # x.append([carbon_flux_mean[i], Sr_ratio[i], Li_isotope[i], Or_isotope[i]])
    #     x.append([carbon_flux_mean[i], carbon_flux_mean[i]*Li_isotope[i], ])

    # reg = linear_model.LinearRegression()
    # reg.fit(x, CO2)
    # print(reg.coef_)
    # print(reg.intercept_)
    # pCO2_pred = reg.predict(x)

    

# =============================================================================
#     # curve fitting (considering the time delay)
#     def func(t, a, b, c, t_delay):
#         
#         F_carbon = np.interp(t+t_delay, Age_flux, carbon_flux_mean)
#         delayed_w_carbon = np.interp(t+t_delay, Age_flux, carbon_flux_mean)
#         F_weathering = np.interp(t, Age_Li, Li_isotope)
#         
#         return a*F_carbon + b*delayed_w_carbon*F_weathering + c
#     
#     popt, pcov = curve_fit(func, Age_CO2, CO2)
#     a_opt, b_opt, c_opt, t_prime_opt = popt
#     print(f'a={a_opt};b={b_opt};c={c_opt};t_delay={t_prime_opt}')
# 
#     pCO2_pred = func(Age_CO2, a_opt, b_opt, c_opt, t_prime_opt)
#     corr_coef, p_value = pearsonr(CO2, pCO2_pred)
#     print(f'correlation coefficient = {corr_coef}; p value = {p_value}\n')
#     
#     with open('fitting_curve.txt', 'w') as f:
#         f.write(f'a={a_opt};b={b_opt};c={c_opt};t_delay={t_prime_opt};correlation coefficient={corr_coef};p value={p_value}\n')
#         for i in range(len(Age_CO2)):
#             f.write(str(Age_CO2[i]) + '\t')
#             f.write('{:.4f}'.format(pCO2_pred[i]) + '\n')
# =============================================================================
     
    
# =============================================================================
#     # curve fitting
#     def func(t, a, b, c):        
#         F_carbon = np.interp(t, Age_flux, carbon_flux_mean)
#         w_carbon = np.interp(t, Age_flux, carbon_flux_mean)
#         F_weathering = np.interp(t, Age_Li, Li_isotope)
#         
#         return a*F_carbon + b*w_carbon*F_weathering + c
#  
#     popt, pcov = curve_fit(func, Age_CO2, CO2)
# 
#     a_opt, b_opt, c_opt = popt
#     #print(f'a={a_opt};b={b_opt};c={c_opt}')
#     pCO2_pred = func(Age_CO2, a_opt, b_opt, c_opt)
#     corr_coef, p_value = pearsonr(CO2, pCO2_pred)
#     print(f'correlation coefficient = {corr_coef}; p value = {p_value}\n')
#     
# =============================================================================

# =============================================================================
#     # curve fitting with constant weight 1 or logarithmic dependence
#     def func(t, a, b, c):        
#         F_carbon = np.interp(t, Age_flux, carbon_flux_mean)
#         F_weathering = np.interp(t, Age_Li, Li_isotope)
#         w_carbon = np.interp(t, Age_flux, R_fs)
#         # w_carbon = 1
#         
#         return a*F_carbon + b*w_carbon*F_weathering + c
#  
#     popt, pcov = curve_fit(func, Age_CO2, CO2)
# 
#     a_opt, b_opt, c_opt = popt
#     print(f'a={a_opt};b={b_opt};c={c_opt}')
#     pCO2_pred = func(Age_CO2, a_opt, b_opt, c_opt)
#     corr_coef, p_value = pearsonr(CO2, pCO2_pred)
#     print(f'correlation coefficient = {corr_coef}; p value = {p_value}\n')
# =============================================================================
    
    
# =============================================================================
#     # 95% credible interval
#     param_samples = np.random.multivariate_normal(popt, pcov, size=1000)
#     y_samples = np.array([func(Age_CO2, *p) for p in param_samples])
#     y_lower, y_upper = np.percentile(y_samples, [2.5, 97.5], axis=0)
# =============================================================================
    
# =============================================================================
#     win_length = 16
#     corr_coef_slide, p_value_slide, Age_slide = slide_correlation(
#         Age_CO2, CO2, pCO2_pred, win_length, plot=True
#         )
#     
#     # output slide window coefficient
#     with open(f'correlation_coefficient_window_{win_length}Myr.txt', 'w') as f:
#         f.write(f'window width = {win_length}\n')
#         for i in range(len(Age_slide)):
#             f.write(str(Age_slide[i]) + '\t')
#             f.write('{:.4f}'.format(corr_coef_slide[i]) + '\t')
#             f.write('{:f}'.format(p_value_slide[i]) + '\n')
# =============================================================================
    


    # multi stage curve fitting
    # stag1: 1-20 Ma
    # stag2: 21-52 Ma
    # stag3: 52-65 Ma
    
    # Stage 1 fitting
    def func(t, a, b, c):        
        F_carbon = np.interp(t, Age_flux[0:20], carbon_flux_mean[0:20])
        w_carbon = np.interp(t, Age_flux[0:20], carbon_flux_mean[0:20])
        F_weathering = np.interp(t, Age_Li[0:20], Li_isotope[0:20])
        
        return a*F_carbon + b*w_carbon*F_weathering + c
 
    popt, pcov = curve_fit(func, Age_CO2[0:20], CO2[0:20])
    a_opt1, b_opt1, c_opt1 = popt
    print(f'Stage1: a={a_opt1}; b={b_opt1}; c={c_opt1}\n')
    
    pCO2_pred_stage1 = func(Age_CO2[0:20], a_opt1, b_opt1, c_opt1)    
    corr_coef1, p_value1 = pearsonr(CO2[0:20], pCO2_pred_stage1)
    print(f'Stage1: correlation coefficient = {corr_coef1}; p value = {p_value1}\n')
    
    # 95% credible interval
    param_samples = np.random.multivariate_normal(popt, pcov, size=1000)
    y_samples = np.array([func(Age_CO2[0:20], *p) for p in param_samples])
    y1_lower, y1_upper = np.percentile(y_samples, [2.5, 97.5], axis=0)
    
    
    # Stage 2 fitting
    def func(t, a, b, c):        
        F_carbon = np.interp(t, Age_flux[20:52], carbon_flux_mean[20:52])
        w_carbon = np.interp(t, Age_flux[20:52], carbon_flux_mean[20:52])
        F_weathering = np.interp(t, Age_Li[20:52], Li_isotope[20:52])
        
        return a*F_carbon + b*w_carbon*F_weathering + c
 
    popt, pcov = curve_fit(func, Age_CO2[20:52], CO2[20:52])
    a_opt2, b_opt2, c_opt2 = popt
    print(f'Stage2: a={a_opt2}; b={b_opt2}; c={c_opt2}\n')
    
    pCO2_pred_stage2 = func(Age_CO2[20:52], a_opt2, b_opt2, c_opt2)    
    corr_coef2, p_value2 = pearsonr(CO2[20:52], pCO2_pred_stage2)
    print(f'Stage2: correlation coefficient = {corr_coef2}; p value = {p_value2}\n')
    
    param_samples = np.random.multivariate_normal(popt, pcov, size=1000)
    y_samples = np.array([func(Age_CO2[20:52], *p) for p in param_samples])
    y2_lower, y2_upper = np.percentile(y_samples, [2.5, 97.5], axis=0)
    
    
    # Stage 3 fitting
    def func(t, a, b, c):        
        F_carbon = np.interp(t, Age_flux[52:], carbon_flux_mean[52:])
        w_carbon = np.interp(t, Age_flux[52:], carbon_flux_mean[52:])
        F_weathering = np.interp(t, Age_Li[52:], Li_isotope[52:])
        
        return a*F_carbon + b*w_carbon*F_weathering + c
 
    popt, pcov = curve_fit(func, Age_CO2[52:], CO2[52:])
    a_opt3, b_opt3, c_opt3 = popt
    print(f'Stage3: a={a_opt3}; b={b_opt3}; c={c_opt3}\n')
    
    pCO2_pred_stage3 = func(Age_CO2[52:], a_opt3, b_opt3, c_opt3)    
    corr_coef3, p_value3 = pearsonr(CO2[52:], pCO2_pred_stage3)
    print(f'Stage3: correlation coefficient = {corr_coef3}; p value = {p_value3}\n')
    
    param_samples = np.random.multivariate_normal(popt, pcov, size=1000)
    y_samples = np.array([func(Age_CO2[52:], *p) for p in param_samples])
    y3_lower, y3_upper = np.percentile(y_samples, [2.5, 97.5], axis=0)
    
    
    # output multi-stage fitting curve
    pCO2_pred_mul = np.concatenate((pCO2_pred_stage1, pCO2_pred_stage2, pCO2_pred_stage3))
    pCO2_pred_lower = np.concatenate((y1_lower, y2_lower, y3_lower))
    pCO2_pred_upper = np.concatenate((y1_upper, y2_upper, y3_upper))
    corr_coef_mul, p_value_mul = pearsonr(CO2, pCO2_pred_mul)
    with open('fitting_curve_multi_stage_CO2.txt', 'w') as f:
        f.write(f'stage 1: a={a_opt1};b={b_opt1}\tc={c_opt1}\tcorrelation coefficient={corr_coef1}\tp value={p_value1}\n')
        f.write(f'stage 2: a={a_opt2};b={b_opt2}\tc={c_opt2}\tcorrelation coefficient={corr_coef2}\tp value={p_value2}\n')
        f.write(f'stage 3: a={a_opt3};b={b_opt3}\tc={c_opt3}\tcorrelation coefficient={corr_coef3}\tp value={p_value3}\n')
        f.write(f'multi-stage: correlation coefficient={corr_coef_mul}\tp value={p_value_mul}\n')
        for i in range(len(Age_CO2)):
            f.write(str(Age_CO2[i]) + '\t')
            f.write('{:.4f}'.format(pCO2_pred_mul[i]) + '\t')
            f.write('{:.4f}'.format(pCO2_pred_lower[i]) + '\t')
            f.write('{:.4f}'.format(pCO2_pred_upper[i]) + '\n')
            
    
# =============================================================================
#     plt.plot(Age_CO2[0:20], pCO2_pred_stage1, label='Stage1', linestyle='-.')
#     plt.plot(Age_CO2[20:52], pCO2_pred_stage2, label='Stage2', linestyle='-.')
#     plt.plot(Age_CO2[52:], pCO2_pred_stage3, label='Stage3', linestyle='-.')
#     plt.show()
# =============================================================================
    

# =============================================================================
#     plt.plot(Age_CO2, CO2,  c='k', linestyle='-', label='Observed')
#     plt.plot(Age_CO2, pCO2_pred, label='original', linestyle='--')
#     plt.show()
# =============================================================================
    
# =============================================================================
#     with open('fitting_curve_wCO2.txt', 'w') as f:
#         f.write(f'a={a_opt};b={b_opt};c={c_opt};correlation coefficient={corr_coef};p value={p_value}\n')
#         for i in range(len(Age_CO2)):
#             f.write(str(Age_CO2[i]) + '\t')
#             f.write('{:.4f}'.format(pCO2_pred[i]) + '\n')
# =============================================================================
# =============================================================================
#             f.write('{:.4f}'.format(y_lower[i]) + '\t')
#             f.write('{:.4f}'.format(y_upper[i]) + '\n')
# =============================================================================
            

        
    
        