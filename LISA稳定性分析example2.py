import numpy as np
a=0;b=1
c=0;d=1
h=1.0/64
h_128=1.0/128
dt=h*0.1
dt_128=h_128*0.1
#空间上一个轴方向的网格数
Number_x=np.int32(1.0/h)
Number_x_128=np.int32(1.0/h_128)
miu=np.zeros((Number_x+1,Number_x+1))
miu_128=np.zeros((Number_x_128+1,Number_x_128+1))
#miu是连续变化的，这里是对系数的确定
#miu=1+1/2*cos(2*pi*x)*cos(2*pi*y)
for i in np.arange(Number_x+1):
    for j in np.arange(Number_x+1):
        miu[i,j]=1+(1.0/2)*np.cos(2*np.pi*j*h)*np.cos(2*np.pi*(1.0-i*h))
#f=10pi*10pi*cos(2pi*x)*cos(2*pi*y)*cos(10*pi*x+1)*cos(10*pi*y+2)*cos(10*pi*sqrt(2)*t+3)
#对于加细网格后系数miu的确定
for i in np.arange(Number_x_128+1):
    for j in np.arange(Number_x_128+1):
        miu_128[i,j]=1+(1.0/2)*np.cos(2*np.pi*j*h_128)*np.cos(2*np.pi*(1.0-i*h_128))
def force(Number_x,h,t):
    f=np.zeros((Number_x+1,Number_x+1))
    for i in np.arange(Number_x+1):
        for j in np.arange(Number_x+1):
            f[i,j]=((100*np.pi**2)*np.cos(2*np.pi*j*h)*np.cos(2*np.pi*(1.0-i*h))*
                    np.cos(10.0*np.pi*j*h+1)*np.cos(10.0*np.pi*(1-i*h)+2)*np.cos(10.0*np.pi*np.sqrt(2)*t+3))
    return f

def u_analytic(x,y,t):
    u=np.cos(10.0*np.pi*x+1.0)*np.cos(10.0*np.pi*y+2.0)*np.cos(10.0*np.pi*np.sqrt(2)*t+3.0)
    return u


def iteration(h,dt,Number_x,miu,u_n,u_n_minus,f,t):
    u_n_plus=np.zeros((Number_x+1,Number_x+1))
    for i in np.arange(1,Number_x):
        for j in np.arange(1,Number_x):
            miu_1=(1.0/4)*(miu[i,j]+miu[i-1,j]+miu[i,j-1]+miu[i-1,j-1])
            miu_2=(1.0/4)*(miu[i,j]+miu[i,j-1]+miu[i+1,j-1]+miu[i+1,j])
            miu_3=(1.0/4)*(miu[i,j]+miu[i+1,j]+miu[i,j+1]+miu[i+1,j+1])
            miu_4=(1.0/4)*(miu[i,j]+miu[i+1,j]+miu[i-1,j+1]+miu[i-1,j])
            alpha=0.25*((1.0/miu_1)+(1.0/miu_2)+(1.0/miu_3)+(1.0/miu_4))
            u_n_plus[i,j]=2*u_n[i,j]-u_n_minus[i,j]+((dt**2)/((h**2)*alpha))*((-4.0)*u_n[i,j]+u_n[i,j+1]+u_n[i-1,j]+u_n[i,j-1]+u_n[i+1,j])+(dt**2)*f[i,j]
    for i in np.arange(Number_x+1):
        u_n_plus[0,i]=np.cos(10.0*np.pi*i*h+1.0)*np.cos(10.0*np.pi*1.0+2.0)*np.cos(10.0*np.pi*np.sqrt(2)*t+3.0)
        u_n_plus[i,0]=np.cos(1.0)*np.cos(10.0*np.pi*(1.0-i*h)+2.0)*np.cos(10.0*np.pi*np.sqrt(2)*t+3.0)
        u_n_plus[Number_x,i]=np.cos(10.0*np.pi*i*h+1.0)*np.cos(2.0)*np.cos(10.0*np.pi*np.sqrt(2)*t+3.0)
        u_n_plus[i,Number_x]=np.cos(10.0*np.pi+1.0)*np.cos(10.0*np.pi*(1.0-i*h)+2.0)*np.cos(10.0*np.pi*np.sqrt(2)*t+3.0)

    return u_n_plus

u=np.zeros((65,65,100))
for i in np.arange(65):
    for j in np.arange(65):
        u[i,j,0]=np.cos(10.0*np.pi*j*h+1.0)*np.cos(10.0*np.pi*(1-i*h)+2.0)*np.cos(3.0)
        u[i,j,1]=np.cos(10.0*np.pi*j*h+1.0)*np.cos(10.0*np.pi*(1-i*h)+2.0)*np.cos(10.0*np.pi*np.sqrt(2.0)*dt+3.0)



u_128=np.zeros((129,129,200))
for i in np.arange(129):
    for j in np.arange(129):
        u_128[i,j,0]=np.cos(10.0*np.pi*j*h_128+1.0)*np.cos(10.0*np.pi*(1-i*h_128)+2.0)*np.cos(3.0)
        u_128[i,j,1]=np.cos(10.0*np.pi*j*h_128+1.0)*np.cos(10.0*np.pi*(1-i*h_128)+2.0)*np.cos(10.0*np.pi*np.sqrt(2.0)*1.0*dt_128+3.0)

for t_n in np.arange(1,99):
    f=force(Number_x,h,(t_n)*dt)
    u[:,:,t_n+1]=iteration(h,dt,Number_x,miu,u[:,:,t_n],u[:,:,t_n-1],f,t_n*dt)

for t_n in np.arange(1,120):
    f=force(Number_x_128,h_128,t_n*dt_128)
    u_128[:,:,t_n+1]=iteration(h_128,dt_128,Number_x_128,miu_128,u_128[:,:,t_n],u_128[:,:,t_n-1],f,t_n*dt_128)


L2_64=0;L2_64_square=0
u_analytic_matrix=np.zeros((65,65))
H1_64=0;H1_64_square=0
for i in np.arange(65):
    for j in np.arange(65):
        L2_64_square=L2_64_square+((u[i,j,50]-u_analytic(j*h,1.0-i*h,50*dt))**2)*(h**2)
        u_analytic_matrix[i,j]=u_analytic(j*h,1.0-i*h,50*dt)
H1_64_A,H1_64_B=np.gradient(u_analytic_matrix-u[:,:,50])
for i in np.arange(65):
    for j in np.arange(65):
        H1_64_square=H1_64_square+H1_64_A[i,j]**2*h**2+H1_64_B[i,j]**2*h**2
L2_64=np.sqrt(L2_64_square)
H1_64_square=H1_64_square+L2_64_square
H1_64=np.sqrt(H1_64_square)
print(L2_64)
print(H1_64)

L2_128=0;L2_128_square=0
u_analytic_matrix_128=np.zeros((129,129))
H1_128=0;H1_128_square=0
for i in np.arange(129):
    for j in np.arange(129):
        L2_128_square=L2_128_square+((u_128[i,j,50]-u_analytic(j*h_128,1.0-i*h_128,50*dt_128))**2)*(h_128**2)
        u_analytic_matrix_128[i,j]=u_analytic(j*h_128,1.0-i*h_128,50*dt_128)
H1_128_A,H1_128_B=np.gradient(u_analytic_matrix_128-u_128[:,:,50])
for i in np.arange(129):
    for j in np.arange(129):
        H1_128_square=H1_128_square+H1_128_A[i,j]**2*h**2+H1_128_B[i,j]**2*h**2
L2_128=np.sqrt(L2_128_square)
H1_128_square=H1_128_square+L2_128_square
H1_128=np.sqrt(H1_128_square)
print(L2_128)
print(H1_128)

L2_128=0;L2_128_square=0
u_analytic_matrix_128=np.zeros((129,129))
H1_128=0;H1_128_square=0
for i in np.arange(129):
    for j in np.arange(129):
        L2_128_square=L2_128_square+((u_128[i,j,100]-u_analytic(j*h_128,1.0-i*h_128,100*dt_128))**2)*(h_128**2)
        u_analytic_matrix_128[i,j]=u_analytic(j*h_128,1.0-i*h_128,100*dt_128)
H1_128_A,H1_128_B=np.gradient(u_analytic_matrix_128-u_128[:,:,100])
for i in np.arange(129):
    for j in np.arange(129):
        H1_128_square=H1_128_square+H1_128_A[i,j]**2*h**2+H1_128_B[i,j]**2*h**2
L2_128=np.sqrt(L2_128_square)
H1_128_square=H1_128_square+L2_128_square
H1_128=np.sqrt(H1_128_square)
print(L2_128)
print(H1_128)