# Numerical-Methods-for-Ultrasonic-Wave-Simulation
This project focuses on the efficient numerical simulation and inversion of acoustic wave propagation. To address the failure of traditional methods in simulating media interfaces, a Local Interaction Simulation Approach (LISA) is introduced, establishing a numerical iterative scheme suitable for media with discontinuous parameters. To achieve efficient computation, GPU parallel acceleration is implemented based on CUDA. Validation through numerical examples demonstrates that this method offers high accuracy and significant acceleration performance. Finally, by embedding the LISA-GPU framework into a full waveform inversion scheme, it provides high-precision and high-efficiency forward modeling support for this nonlinear inverse problem.


Main results:
The numerical accuracy and computational performance of the LISA scheme
are evaluated by comparing the numerical solution with the analytical
solution under both CPU and GPU implementations.

As shown in the table below, the numerical results demonstrate a highly
satisfactory agreement with the analytical solution in both the $L_2$
and $H_1$ norms. The slight discrepancies between CPU and GPU results are
mainly caused by differences in floating-point precision. In particular,
GPU computations typically use single-precision arithmetic, while CPU
computations are generally performed in double precision.

For a spatial grid resolution of $64 \times 64$, the computational times
of CPU and GPU implementations are comparable. This is largely due to
the highly optimized numerical operations provided by the NumPy
library on the CPU side, which limits the performance advantage of GPU
acceleration at relatively coarse grid resolutions.

| Device | $L_2$ Error | Relative $L_2$ Error | $H_1$ Error | Time (s) |
| ------ | ----------- | -------------------- | ----------- | -------- |
| CPU    | 2.0766e-02  | 4.7437e-02           | 2.5249e-02  | 5.2425   |
| GPU    | 2.0766e-02  | 4.7437e-02           | 2.5249e-02  | 5.3657   |


When the spatial step size is set to h = 1/1024, the time step is chosen
as Δt = 0.1h in order to ensure numerical stability. The errors in the
L2 and H1 norms are evaluated at the 1600th time step. The corresponding
physical time t is identical to the final time obtained after 100 time
steps on the 64 × 64 grid.

Using GPU computation, the error in the L2 norm is approximately
0.0014704622931797885, with a relative error of
0.003381972106473233.

The error measured in the H1 norm is approximately
0.0015023849514756457, with a relative error of
0.003452142107724719.

The total computational time is approximately
24.95804786682129 s.

Notably, when the total number of grid points is increased by a factor
of 16³, the computational time increases by only about five
times, while maintaining a highly satisfactory level of accuracy.
This result clearly demonstrates the significant advantages of
parallel computation for large-scale numerical simulations.
