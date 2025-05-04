import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.stats import pearsonr
import math

data_oryz = np.genfromtxt('OryzData.csv', delimiter=',', skip_header=1)
data_rhyz = np.genfromtxt('RhyzData.csv', delimiter=',', skip_header=1)
data_rhyzoryz = np.genfromtxt('RhyzOryzData.csv', delimiter=',', skip_header=1)


t_oryz_data = data_oryz[:, 0]
m_oryz_data = data_oryz[:, 1]
t_rhyz_data = data_rhyz[:, 0]
m_rhyz_data = data_rhyz[:, 1]
t_shared = data_rhyzoryz[:, 0]
mrhyz_rhyzoryz_data = data_rhyzoryz[:, 1]
moryz_rhyzoryz_data = data_rhyzoryz[:, 2]

plt.scatter(t_oryz_data, m_oryz_data, label="Oryzaephilus Population (No competition)", color="yellowgreen")
plt.scatter(t_rhyz_data, m_rhyz_data, label="Rhyzopertha Population (No competition)", color="darkolivegreen")
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.show()

plt.scatter(t_shared, mrhyz_rhyzoryz_data, label='Rhyzopertha Population (Competition)',color = "indianred")
plt.scatter(t_shared, moryz_rhyzoryz_data, label='Oryzaephilus Population (Competition)',color = "maroon")
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.show()

oryz_0 = m_oryz_data[0]
r_0_oryz = 0.05
K_0_oryz = 450

t_oryz_min = t_oryz_data[0]
t_oryz_max = t_oryz_data[-1]

def logistic_oryz(t, x,r, K):
    return r * x * (1 - x/K)

def logisticModelOryz(t, r,K):
    sol = solve_ivp(logistic_oryz, [t_oryz_min, t_oryz_max], [oryz_0], args=(r, K), t_eval=t)
    return sol.y[0]

def logisticModelOryz_objective_function(params, t, y):
    r,K = params
    y_pred = logisticModelOryz(t, r,K)
    return np.sum((y_pred - y) ** 2)

logisticOryzResult = minimize(logisticModelOryz_objective_function, [r_0_oryz,K_0_oryz], args=(t_oryz_data, m_oryz_data), method='Nelder-Mead')

logisticOryz_r_est, logisticOryz_K_est = logisticOryzResult.x
print("Logistic: r =", logisticOryz_r_est, "K =", logisticOryz_K_est)

t_oryz_model = np.linspace(t_oryz_min, t_oryz_max, 1000)
m_oryz_fit = logisticModelOryz(t_oryz_model, logisticOryz_r_est,logisticOryz_K_est)

plt.scatter(t_oryz_data, m_oryz_data, label="Data", color="black")
plt.plot(t_oryz_model, m_oryz_fit, label="Oryzaephilus Fit", color="blue")
plt.xlabel('Time (days)')
plt.ylabel('Oryzaephilus population')
plt.legend()
plt.show()

rhyz_0 = m_rhyz_data[0]
r_0_rhyz = 0.05
K_0_rhyz = 450

t_rhyz_min = t_rhyz_data[0]
t_rhyz_max = t_rhyz_data[-1]

def logistic_rhyz(t, x,r, K):
    return r * x * (1 - x/K)

def logisticModelrhyz(t, r,K):
    sol = solve_ivp(logistic_rhyz, [t_rhyz_min, t_rhyz_max], [rhyz_0], args=(r, K), t_eval=t)
    return sol.y[0]

def logisticModelrhyz_objective_function(params, t, y):
    r,K = params
    y_pred = logisticModelrhyz(t, r,K)
    return np.sum((y_pred - y) ** 2)

logisticrhyzResult = minimize(logisticModelrhyz_objective_function, [r_0_rhyz,K_0_rhyz], args=(t_rhyz_data, m_rhyz_data), method='Nelder-Mead')

logisticrhyz_r_est, logisticrhyz_K_est = logisticrhyzResult.x
print("Logistic: r =", logisticrhyz_r_est, "K =", logisticrhyz_K_est)

t_rhyz_model = np.linspace(t_rhyz_min, t_rhyz_max, 1000)
m_rhyz_fit = logisticModelrhyz(t_rhyz_model, logisticrhyz_r_est,logisticrhyz_K_est)

plt.scatter(t_rhyz_data, m_rhyz_data, label="Data", color="black")
plt.plot(t_rhyz_model, m_rhyz_fit, label="Rhyzopertha Fit", color="blue")
plt.xlabel('Time (days)')
plt.ylabel('Rhyzopertha population')
plt.legend()
plt.show()


oryz_fit_interp = np.interp(t_oryz_data, t_oryz_model, m_oryz_fit)

ss_res_oryz = np.sum((m_oryz_data - oryz_fit_interp) ** 2)
ss_tot_oryz = np.sum((m_oryz_data - np.mean(m_oryz_data)) ** 2)
r2_oryz = 1 - (ss_res_oryz / ss_tot_oryz)

print(f"R^2 for Oryzaephilus logistic model: {r2_oryz:.4f}")

rhyz_fit_interp = np.interp(t_rhyz_data, t_rhyz_model, m_rhyz_fit)

ss_res_rhyz = np.sum((m_rhyz_data - rhyz_fit_interp) ** 2)
ss_tot_rhyz = np.sum((m_rhyz_data - np.mean(m_rhyz_data)) ** 2)
r2_rhyz = 1 - (ss_res_rhyz / ss_tot_rhyz)

print(f"R^2 for Rhyzopertha logistic model: {r2_rhyz:.4f}")



def competition_system(t, Z, r_R, K_R, a_R, r_O, K_O, a_O):
    X, Y = Z
    dXdt = r_R * X * (1 - X / K_R) - a_R * X * Y
    dYdt = r_O * Y * (1 - Y / K_O) - a_O * X * Y
    return [dXdt, dYdt]

t_shared = data_rhyzoryz[:, 0]
t_shared_min = t_shared[0]
t_shared_max = t_shared[-1]
mrhyz_rhyzoryz_data = data_rhyzoryz[:, 1]
moryz_rhyzoryz_data = data_rhyzoryz[:, 2]

X0 = mrhyz_rhyzoryz_data[0]
Y0 = moryz_rhyzoryz_data[0]

r0_X = 0.05
K0_X = 350
r0_Y = 0.05
K0_Y = 500

oryz_r = logisticOryz_r_est 
oryz_K = logisticOryz_K_est

rhyz_r = logisticrhyz_r_est 
rhyz_K = logisticrhyz_K_est

def competition_system(t, Z, a_X, a_Y):
    X, Y = Z
    dXdt = rhyz_r * X * (1 - X / rhyz_K) - a_X * X * Y
    dYdt = oryz_r * Y * (1 - Y / oryz_K) - a_Y * X * Y
    return [dXdt, dYdt]

def competition_model(t, a_X,a_Y):
    sol = solve_ivp(competition_system, [t_shared_min, t_shared_max], [X0,Y0], args=(a_X,a_Y), t_eval=t)
    return sol.y

def competition_objective(params, t, rhyz_real, oryz_real):
    a_X, a_Y = params
    X_pred, Y_pred = competition_model(t, a_X, a_Y)
    error_X = np.sum((X_pred - rhyz_real) ** 2) / len(rhyz_real)
    error_Y = np.sum((Y_pred - oryz_real) ** 2) / len(oryz_real)
    return error_X + error_Y

a0_X = 0.0001
a0_Y = 0.0001
params0 = [a0_X, a0_Y]

result = minimize(
    competition_objective,
    params0,
    args=(t_shared, mrhyz_rhyzoryz_data, moryz_rhyzoryz_data),
    method='Nelder-Mead'
)

a_X_fit, a_Y_fit = result.x
print("Fitted Competition Coefficients:")
print(f"a_X = {a_X_fit}")
print(f"a_Y = {a_Y_fit}")

t_model = np.linspace(t_shared[0], t_shared[-1], 1000)
X_model, Y_model = competition_model(t_model, a_X_fit, a_Y_fit)

plt.figure(figsize=(8, 5))
plt.scatter(t_shared, mrhyz_rhyzoryz_data, label='Rhyzopertha Data',color = "indianred")
plt.scatter(t_shared, moryz_rhyzoryz_data, label='Oryzaephilus Data',color = "mediumseagreen")
plt.plot(t_model, X_model, label='Rhyzopertha Fit', color = "maroon")
plt.plot(t_model, Y_model, label='Oryzaephilus Fit',color="darkgreen")
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.show()

#Code for R^2 from ChatGPT
X_pred_interp = np.interp(t_shared, t_model, X_model)
Y_pred_interp = np.interp(t_shared, t_model, Y_model)
ss_res_X = np.sum((mrhyz_rhyzoryz_data - X_pred_interp) ** 2)
ss_tot_X = np.sum((mrhyz_rhyzoryz_data - np.mean(mrhyz_rhyzoryz_data)) ** 2)
r2_X = 1 - ss_res_X / ss_tot_X
ss_res_Y = np.sum((moryz_rhyzoryz_data - Y_pred_interp) ** 2)
ss_tot_Y = np.sum((moryz_rhyzoryz_data - np.mean(moryz_rhyzoryz_data)) ** 2)
r2_Y = 1 - ss_res_Y / ss_tot_Y

print(f"R^2 for Rhyzopertha: {r2_X:.4f}")
print(f"R^2 for Oryzaephilus: {r2_Y:.4f}")


def localSensitivity(func, t, est, param1, param2):
    y0 = func(t, (1 - h) * est, param1, param2)
    y1 = func(t, est, param1, param2)
    y2 = func(t, (1 + h) * est, param1, param2)
    return (y2 - y0) / (2 * h * est) * (est / y1)

def competition_system_with_const(t, Z,rhyz_param_r,rhyz_param_K,rhyz_param_a):
    X, Y = Z
    dXdt = rhyz_param_r * X * (1 - X / rhyz_param_K) - rhyz_param_a * X * Y
    dYdt = oryz_r * Y * (1 - Y / oryz_K) - a_Y_fit * X * Y
    return [dXdt, dYdt]

def model_rR (t,param_r,K,a):
    sol = solve_ivp(competition_system_with_const, [t_shared_min, t_shared_max], [X0,Y0], args=(param_r,K,a), t_eval=t)
    return sol.y[0]

def model_KR (t,param_K,r,a):
    sol = solve_ivp(competition_system_with_const, [t_shared_min, t_shared_max], [X0,Y0], args=(r,param_K,a), t_eval=t)
    return sol.y[0]

def model_aR (t,param_a,r,K):
    sol = solve_ivp(competition_system_with_const, [t_shared_min, t_shared_max], [X0,Y0], args=(r,K,param_a), t_eval=t)
    return sol.y[0]

param_r = logisticrhyz_r_est
param_K = logisticrhyz_K_est
param_a = a_X_fit
h = 0.01

S_r = localSensitivity(model_rR, t_shared, param_r, param_K, param_a)
S_K = localSensitivity(model_KR, t_shared, param_K,param_r, param_a)
S_a = localSensitivity(model_aR, t_shared, param_a,param_r, param_K)

plt.plot(t_shared,S_r,label="Sensitivity of intrinsic growth rate (r)")
plt.plot(t_shared,S_K,label="Sensitivity of carrying capacity (K)")
plt.plot(t_shared,S_a,label="Sensitivity of competition coefficient (a)")
plt.xlabel('time (days)')
plt.ylabel('sensitivity')
plt.legend()
plt.show()


corr,pval = pearsonr(S_r,S_K)
R_sq = corr**2
# identifiability: plot sensitivity relationships
plt.plot(S_r,S_K)
plt.xlabel('S_r')
plt.ylabel('S_K')
# label plot with coefficient of determination
plt.title(f"R^2 = {R_sq:.2f}") 
plt.show()

corr,pval = pearsonr(S_r,S_a)
R_sq = corr**2
# identifiability: plot sensitivity relationships
plt.plot(S_a,S_r)
plt.xlabel('S_r')
plt.ylabel('S_a')
# label plot with coefficient of determination
plt.title(f"R^2 = {R_sq:.2f}") 
plt.show()

corr,pval = pearsonr(S_K,S_a)
R_sq = corr**2
# identifiability: plot sensitivity relationships
plt.plot(S_K,S_a)
plt.xlabel('S_K')
plt.ylabel('S_a')
# label plot with coefficient of determination
plt.title(f"R^2 = {R_sq:.2f}") 
plt.show()

n= len(mrhyz_rhyzoryz_data)
def AAIC_competition_func(func,param_array):
    return n*math.log(func(param_array, t_shared, mrhyz_rhyzoryz_data,moryz_rhyzoryz_data)/n)+2*(len(param_array)) + 2*len(param_array)*(len(param_array)+1)/(n-len(param_array)-1)

AAIC_competition = AAIC_competition_func(competition_objective,[a_X_fit,a_Y_fit])
print("AAIC Lotka Volterra: " ,AAIC_competition)

def logisticModelOryz_AAIC_FIT(t, r,K):
    sol = solve_ivp(logistic_oryz, [t_shared_min, t_shared_max], [moryz_rhyzoryz_data[0]], args=(r, K), t_eval=t)
    return sol.y[0]

def logisticModelOryz_objective_function(params, t, y):
    r,K = params
    y_pred = logisticModelOryz_AAIC_FIT(t, r,K)
    return np.sum((y_pred - y) ** 2)

def logisticModelRhyz_AAIC_FIT(t, r,K):
    sol = solve_ivp(logistic_rhyz, [t_shared_min, t_shared_max], [mrhyz_rhyzoryz_data[0]], args=(r, K), t_eval=t)
    return sol.y[0]

def logisticModelRhyz_objective_function(params, t, y):
    r,K = params
    y_pred = logisticModelRhyz_AAIC_FIT(t, r,K)
    return np.sum((y_pred - y) ** 2)

def AAIC_logistic_func(func,param_array,t_data,m_data):
    return n*math.log(func(param_array, t_data,m_data)/n)+2*(len(param_array)) + 2*len(param_array)*(len(param_array)+1)/(n-len(param_array)-1)

AAIC_logistic_oryz = AAIC_logistic_func(logisticModelOryz_objective_function,[logisticOryz_r_est,logisticOryz_K_est],t_shared,moryz_rhyzoryz_data)
AAIC_logistic_rhyz = AAIC_logistic_func(logisticModelRhyz_objective_function,[logisticrhyz_r_est,logisticrhyz_K_est],t_shared,mrhyz_rhyzoryz_data)

print("AAIC Oryz: " ,AAIC_logistic_oryz)
print("AAIC Rhyz: " ,AAIC_logistic_rhyz)