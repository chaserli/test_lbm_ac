import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt

NU = 0.01 / np.pi
N = 400
L = 2.
T = 1

x = np.linspace(-L/2, L/2, N, endpoint=False)
DX = L/N
k = fft.rfftfreq(N, d=DX) * 2 * np.pi
u0 = -np.sin(np.pi * x)
DT = min(T/200,0.8*DX**2/NU)
t = np.arange(0,int(T/DT +1))*DT

# 3/2 dealiasing
def burgers_rhs(t,u):
    u_ext = fft.irfft(u,n=int(N*3/2))
    dudx_ext = fft.irfft(1j * k * u,n=int(N*3/2))
    u_dudx = fft.rfft(u_ext * dudx_ext)[:N//2+1]*3/2
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
