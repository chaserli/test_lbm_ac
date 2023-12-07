import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

NU = 0.01 / np.pi
N = 512
L = 2
T = 1

x = np.linspace(-L/2, L/2, N, endpoint=False)
k = fft.rfftfreq(N, L / N) * 2 * np.pi
u0 = -np.sin(np.pi * x)
t = np.linspace(0, T, 200)

# TODO: 3/2 dealiasing
def burgers_rhs(t,u):
    u_ext = fft.irfft(u)
    dudx_ext = fft.irfft(1j * k * u)
    u_dudx = fft.rfft(u_ext * dudx_ext)
    return -NU * k**2 * u - u_dudx

sols=np.zeros((t.size,x.size))
from scipy.integrate import solve_ivp
ode_sol = solve_ivp(burgers_rhs, t_span=(0,T), y0=fft.rfft(u0), method='RK45', t_eval=t)
for i in range(0, t.size):
    sols[i]=fft.irfft(ode_sol.y[:, i])

if __name__=='__main__':
    plt.figure()
    plt.pcolor(t,x,sols.T,cmap='jet')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.colorbar()
    plt.show()
