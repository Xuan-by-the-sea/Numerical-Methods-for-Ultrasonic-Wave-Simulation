import numpy as np
import math as m
import matplotlib.pyplot as plt
import copy
import scipy.sparse          # 稀疏矩阵
import scipy.signal as sig   # 信号处理
from numba import cuda
import time
a=-0.5;b=1.5;c=-0.5;d=1.5;T=3
Lx=b-a;Ly=d-c
Nx=800;Ny=800                         # x、y 轴上的剖分数
Nt=1200                                # 时间步数
x=np.linspace(a,b,Nx+1)               # 网格点的 x 坐标
y=np.linspace(c,d,Ny+1)               # 网格点的 y 坐标
dx=x[1]-x[0]                          # x 轴上的步长    
dy=y[1]-y[0]                          # y 轴上的步长
lamda_left=4.4;mu_left= 2.09;rho_left=1.2  # 有机玻璃 Plexiglas
lamda_right=56.0;mu_right=26.0;rho_right=2.7  # 铝 AI
lamda_0 = 0.05; c_p = m.sqrt( (lamda_left + 2.0 * mu_left)/rho_left ); f_c = c_p/lamda_0
N = 10; Peridic = N/f_c       
#dt = Peridic/(20*N)  # 20N points in one time frequency domain wavelegth
dt = Peridic/(60-1)/(Nx/((b-a)*100))/(N/3)
CFL = c_p*dt/dx       # If dt is not divide Nx/100, the scheme is not stable!
t = np.linspace(0,Nt*dt,Nt+1)        # t-coordinate of mesh points
dt2 = (dt)**2
dt2dx2 = (dt/dx)**2                  # help coefficients 

sigma_1_left=lamda_left+2*mu_left;sigma_1_right=lamda_right+2*mu_right 
sigma_2_left=lamda_left+2*mu_left;sigma_2_right=lamda_right+2*mu_right
nu_left=lamda_left+mu_left;nu_right=lamda_right+mu_right
g_left=(lamda_left-mu_left)/2.0;g_right=(lamda_right-mu_right)/2.0
rho=(rho_left+rho_right)/2.0
mu=(mu_left+mu_right)/2.0
sigma_1=(sigma_1_left+sigma_1_right)/2.0
sigma_2=(sigma_2_left+sigma_2_right)/2.0
cpu_para=np.array([Nx,mu_left,rho_left,mu_right,rho_right,dt,dt2,dt2dx2,sigma_1_left,sigma_1_right,sigma_2_left,\
                       sigma_2_right,nu_left,nu_right,g_left,g_right,rho,mu,sigma_1,sigma_2])

Ix_L=range(0,int(Nx/2+1))   # 0,1,...,Nx/2
Ix_R=range(int(Nx/2),Nx+1)  # Nx/2,Nx/2+1,...,Nx
#Ix=range(0,Nx+1)
Iy=range(0,Ny+1)       # 0,1,...,Ny
It=range(1,Nt+1)       # 1,2,...,Nt
def g(x,x_0,sigma): 
    g = (1.0/((m.sqrt(2.0*m.pi))*sigma))*m.exp(-((x-x_0)**2)/(2.0*(sigma**2)))
    return g

def f(t,f_c,N):
    f = (1.0/2.0)*(1.0-np.cos(2.0*np.pi*f_c*t/N))*np.sin(2.0*np.pi*f_c*t)
    return f

F = np.zeros((Nx+1,Ny+1))
for i in range(int(Nx/8-10),int(Nx/8+11)):   # 以Nx/4为中心的21个点：90 …… 110
    F[int(Nx/4+i),int((3*Nx/4)-(Nx/4+i))]=g(-0.5+(Nx/4+i)*dx,-0.5+3*Nx/8*dx,lamda_0/2)*m.sqrt(2)/2

Dampu_cpu=np.zeros((Nx+1,Ny+1))
Dampv_cpu=np.zeros((Nx+1,Ny+1))
damp=1000
for i in range(0,int(Nx/4)):
    Dampv_cpu[:,i]=damp*np.ones(Nx+1)*((Nx/4-i)*dx)**3

for i in range(int(3*Nx/4+1),int(Nx+1)):
    Dampv_cpu[:,i]=damp*np.ones(Nx+1)*((i-3*Nx/4)*dx)**3

for i in range(0,int(Nx/4)):
    Dampu_cpu[i,:]=damp*np.ones(Nx+1)*((Nx/4-i)*dx)**3

for i in range(int(3*Nx/4+1),int(Nx+1)):
    Dampu_cpu[i,:]=damp*np.ones(Nx+1)*((i-3*Nx/4)*dx)**3

Dampu=cuda.to_device(Dampu_cpu)
Dampv=cuda.to_device(Dampv_cpu)
para=cuda.to_device(cpu_para)

@cuda.jit
def gpu_loop_nof_force(para,gpu_un,gpu_vn,u_0,u_1,v_0,v_1,G):
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x+1
    j = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y+1
    if i<int(para[0]/2) and j<para[0]:
        gpu_un[i,j]=2.0*u_1[i,j]-u_0[i,j]+(1.0/para[2])*para[7]*\
                    (para[8]*u_1[i+1,j]+para[8]*u_1[i-1,j]+para[1]*(u_1[i,j+1]+u_1[i,j-1])-2.0*(para[8]+para[1])*u_1[i,j]\
                     +(1.0/4.0)*(para[12]*(v_1[i+1,j+1]-v_1[i+1,j-1])+para[12]*(v_1[i-1,j-1]-v_1[i-1,j+1])\
                                 ))+0.1*para[6]*G[i,j] 
        gpu_vn[i,j]=2.0*v_1[i,j]-v_0[i,j]+(1.0/para[2])*para[7]*\
                    (para[1]*v_1[i+1,j]+para[1]*v_1[i-1,j]+para[10]*(v_1[i,j+1]+v_1[i,j-1])-2.0*(para[10]+para[1])*v_1[i,j]\
                     +(1.0/4.0)*(para[12]*(u_1[i+1,j+1]-u_1[i+1,j-1])+para[12]*(u_1[i-1,j-1]-u_1[i-1,j+1])\
                                 ))+0.1*para[6]*G[i,j]
    
    if i==int(para[0]/2) and j<para[0]:
        gpu_un[i,j]=2.0*u_1[i,j]-u_0[i,j]+(1.0/para[16])*para[7]*\
                    (para[9]*u_1[i+1,j]+para[8]*u_1[i-1,j]+para[17]*(u_1[i,j+1]+u_1[i,j-1])-2.0*(para[18]+para[17])*u_1[i,j]\
                     +(1.0/4.0)*(para[13]*(v_1[i+1,j+1]-v_1[i+1,j-1])+para[12]*(v_1[i-1,j-1]-v_1[i-1,j+1])\
                                 +2.0*(para[15]-para[14])*(v_1[i,j+1]-v_1[i,j-1])))+0.1*para[6]*G[i,j]
        gpu_vn[i,j]=2.0*v_1[i,j]-v_0[i,j]+(1.0/para[16])*para[7]*\
                    (para[3]*v_1[i+1,j]+para[1]*v_1[i-1,j]+para[19]*(v_1[i,j+1]+v_1[i,j-1])-2.0*(para[19]+para[17])*v_1[i,j]\
                     +(1.0/4.0)*(para[13]*(u_1[i+1,j+1]-u_1[i+1,j-1])+para[12]*(u_1[i-1,j-1]-u_1[i-1,j+1])\
                                 -2.0*(para[15]-para[14])*(u_1[i,j+1]-u_1[i,j-1])))+0.1*para[6]*G[i,j]

    if i>int(para[0]/2) and i<para[0] and j<para[0]:
        gpu_un[i,j]=2.0*u_1[i,j]-u_0[i,j]+(1.0/para[4])*para[7]*\
                    (para[9]*u_1[i+1,j]+para[9]*u_1[i-1,j]+para[3]*(u_1[i,j+1]+u_1[i,j-1])-2.0*(para[9]+para[3])*u_1[i,j]\
                     +(1.0/4.0)*(para[13]*(v_1[i+1,j+1]-v_1[i+1,j-1])+para[13]*(v_1[i-1,j-1]-v_1[i-1,j+1])\
                                 ))+0.1*para[6]*G[i,j]
        gpu_vn[i,j]=2.0*v_1[i,j]-v_0[i,j]+(1.0/para[4])*para[7]*\
                    (para[3]*v_1[i+1,j]+para[3]*v_1[i-1,j]+para[11]*(v_1[i,j+1]+v_1[i,j-1])-2.0*(para[11]+para[3])*v_1[i,j]\
                     +(1.0/4.0)*(para[13]*(u_1[i+1,j+1]-u_1[i+1,j-1])+para[13]*(u_1[i-1,j-1]-u_1[i-1,j+1])\
                                 ))+0.1*para[6]*G[i,j]

@cuda.jit
def gpu_loop_nof(para,gpu_un,gpu_vn,u_0,u_1,v_0,v_1,Dampu,Dampv):
    i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x+1
    j = cuda.threadIdx.y + cuda.blockDim.y * cuda.blockIdx.y+1
    if i<int(para[0]/2) and j<para[0]:
        gpu_un[i,j]=(2.0*u_1[i,j]-u_0[i,j]+para[5]*Dampu[i,j]*u_0[i,j]/2+(1.0/para[2])*para[7]*\
                    (para[8]*u_1[i+1,j]+para[8]*u_1[i-1,j]+para[1]*(u_1[i,j+1]+u_1[i,j-1])-2.0*(para[8]+para[1])*u_1[i,j]\
                     +(1.0/4.0)*(para[12]*(v_1[i+1,j+1]-v_1[i+1,j-1])+para[12]*(v_1[i-1,j-1]-v_1[i-1,j+1])\
                                 )))/(para[5]*Dampu[i,j]/2+1) 
        gpu_vn[i,j]=(2.0*v_1[i,j]-v_0[i,j]+para[5]*Dampv[i,j]*v_0[i,j]/2+(1.0/para[2])*para[7]*\
                    (para[1]*v_1[i+1,j]+para[1]*v_1[i-1,j]+para[10]*(v_1[i,j+1]+v_1[i,j-1])-2.0*(para[10]+para[1])*v_1[i,j]\
                     +(1.0/4.0)*(para[12]*(u_1[i+1,j+1]-u_1[i+1,j-1])+para[12]*(u_1[i-1,j-1]-u_1[i-1,j+1])\
                                 )))/(para[5]*Dampv[i,j]/2+1)
    
    if i==int(para[0]/2) and j<para[0]:
        gpu_un[i,j]=(2.0*u_1[i,j]-u_0[i,j]+para[5]*Dampu[i,j]*u_0[i,j]/2+(1.0/para[16])*para[7]*\
                    (para[9]*u_1[i+1,j]+para[8]*u_1[i-1,j]+para[17]*(u_1[i,j+1]+u_1[i,j-1])-2.0*(para[18]+para[17])*u_1[i,j]\
                     +(1.0/4.0)*(para[13]*(v_1[i+1,j+1]-v_1[i+1,j-1])+para[12]*(v_1[i-1,j-1]-v_1[i-1,j+1])\
                                 +2.0*(para[15]-para[14])*(v_1[i,j+1]-v_1[i,j-1]))))/(para[5]*Dampu[i,j]/2+1)  
        gpu_vn[i,j]=(2.0*v_1[i,j]-v_0[i,j]+para[5]*Dampv[i,j]*v_0[i,j]/2+(1.0/para[16])*para[7]*\
                    (para[3]*v_1[i+1,j]+para[1]*v_1[i-1,j]+para[19]*(v_1[i,j+1]+v_1[i,j-1])-2.0*(para[19]+para[17])*v_1[i,j]\
                     +(1.0/4.0)*(para[13]*(u_1[i+1,j+1]-u_1[i+1,j-1])+para[12]*(u_1[i-1,j-1]-u_1[i-1,j+1])\
                                 -2.0*(para[15]-para[14])*(u_1[i,j+1]-u_1[i,j-1]))))/(para[5]*Dampv[i,j]/2+1)

    if i>int(para[0]/2) and i<para[0] and j<para[0]:
        gpu_un[i,j]=(2.0*u_1[i,j]-u_0[i,j]+para[5]*Dampu[i,j]*u_0[i,j]/2+(1.0/para[4])*para[7]*\
                    (para[9]*u_1[i+1,j]+para[9]*u_1[i-1,j]+para[3]*(u_1[i,j+1]+u_1[i,j-1])-2.0*(para[9]+para[3])*u_1[i,j]\
                     +(1.0/4.0)*(para[13]*(v_1[i+1,j+1]-v_1[i+1,j-1])+para[13]*(v_1[i-1,j-1]-v_1[i-1,j+1])\
                                 )))/(para[5]*Dampu[i,j]/2+1) 
        gpu_vn[i,j]=(2.0*v_1[i,j]-v_0[i,j]+para[5]*Dampv[i,j]*v_0[i,j]/2+(1.0/para[4])*para[7]*\
                    (para[3]*v_1[i+1,j]+para[3]*v_1[i-1,j]+para[11]*(v_1[i,j+1]+v_1[i,j-1])-2.0*(para[11]+para[3])*v_1[i,j]\
                     +(1.0/4.0)*(para[13]*(u_1[i+1,j+1]-u_1[i+1,j-1])+para[13]*(u_1[i-1,j-1]-u_1[i-1,j+1])\
                                 )))/(para[5]*Dampv[i,j]/2+1)

threads_per_block = (16, 16)
blocks_per_grid_x = int(m.ceil((Nx+1)/threads_per_block[0]))
blocks_per_grid_y = int(m.ceil((Nx+1)/threads_per_block[1]))
blocksPerGrid = (blocks_per_grid_x, blocks_per_grid_y)
gpu_un=cuda.device_array((Nx+1,Nx+1))
gpu_vn=cuda.device_array((Nx+1,Nx+1))

'''u_0_device=cuda.device_array((Nx+1,Ny+1))          #initial solution array in time step 0
u_1_device=cuda.device_array((Nx+1,Ny+1))          #initial solution array in time step 1
v_0_device=cuda.device_array((Nx+1,Ny+1))          #initial solution array in time step 0
v_1_device=cuda.device_array((Nx+1,Ny+1))          #initial solution array in time step 1'''

u_all=cuda.device_array((Nx+1,2*(Ny+1)))
v_all=cuda.device_array((Nx+1,2*(Ny+1)))
start=time.time()
for n in It[1:200]: #loop for t = 1,2,..,Nt
    G=f((n-1)*dt*(Nx/2/100),f_c,N)*F
    G_device=cuda.to_device(G)
    gpu_loop_nof_force[blocksPerGrid, threads_per_block](para,gpu_un,gpu_vn,u_all[:,0:Ny+1],u_all[:,Ny+1:2*(Ny+1)],\
                                                         v_all[:,0:Ny+1],v_all[:,Ny+1:2*(Ny+1)],G_device)
                                                         
    u_all[:,0:Ny+1]=u_all[:,Ny+1:2*(Ny+1)]
    u_all[:,Ny+1:2*(Ny+1)]=gpu_un
    v_all[:,0:Ny+1]=v_all[:,Ny+1:2*(Ny+1)]
    v_all[:,Ny+1:2*(Ny+1)]=gpu_vn
    u=gpu_un.copy_to_host()
    v=gpu_vn.copy_to_host()
    U=np.sqrt(u**2+v**2)
    # print(U[Nx,Nx])
    #if n==4:
    #    print(np.max(U))

for n in It[200:Nt]: #loop for t = 1,2,..,Nt
    gpu_loop_nof[blocksPerGrid,threads_per_block](para,gpu_un,gpu_vn,u_all[:,0:Ny+1],u_all[:,Ny+1:2*(Ny+1)],\
                                                         v_all[:,0:Ny+1],v_all[:,Ny+1:2*(Ny+1)],Dampu,Dampv)

    u_all[:,0:Ny+1]=u_all[:,Ny+1:2*(Ny+1)]
    u_all[:,Ny+1:2*(Ny+1)]=gpu_un
    v_all[:,0:Ny+1]=v_all[:,Ny+1:2*(Ny+1)]
    v_all[:,Ny+1:2*(Ny+1)]=gpu_vn
    u=gpu_un.copy_to_host()
    v=gpu_vn.copy_to_host()
    U=np.sqrt(u**2+v**2)
x_point_mesh, y_point_mesh = np.meshgrid(x, y)
# d3为添加的子图
d3 = plt.axes(projection='3d')
d3.plot_surface(x_point_mesh, y_point_mesh, U, cmap='rainbow')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
d3.set_zlabel('x(位置)/m')
d3.set_xlabel('y(位置)/m')
d3.set_ylabel('A(振幅)/s')
d3.set_title('最终时刻波的振幅大小分布')
plt.show()
end=time.time()
print(end-start)
# print(U[Nx,Nx])
#print(np.max(U))


