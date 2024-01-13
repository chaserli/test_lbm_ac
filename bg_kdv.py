import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from tqdm import trange

N = 256
L = 24
T = 1

x = np.linspace(0, L, N, endpoint=False)
DX = L/N
k = fft.rfftfreq(N, d=DX) * 2 * np.pi

N_KDV = 4
X0_KDV = 3
u0 = N_KDV*(N_KDV+1)/(np.cosh(x-X0_KDV)**2)
DT = min(0.05*DX**3,DX/(6*np.abs(u0).max()))
NT = int(T/DT)+1

def kdv_rhs(u):
    u_ext = fft.irfft(u,n=int(N*3/2),workers=-1)
    u2 = fft.rfft(u_ext **2,workers=-1)[:N//2+1]*3/2
    return -3j*k* u2 - (1j*k)**3*u

sols=np.zeros((NT,x.size))
u_hat=fft.rfft(u0)
for i in trange(NT):
    k1 = kdv_rhs(u_hat)
    k2 = kdv_rhs(u_hat+DT/2*k1)
    k3 = kdv_rhs(u_hat+DT/2*k2)
    k4 = kdv_rhs(u_hat+DT*k3)
    u_hat += DT/6*(k1+2*k2+2*k3+k4)
    sols[i]=fft.irfft(u_hat)

if __name__=='__main__':
    plt.figure()
    plt.pcolor(np.arange(1,NT+1)*DT,x,sols.T,cmap='jet')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.plot(x,sols[-40])
    plt.plot(x,sols[-1])
    plt.show()
