# SteadyTimeSettings

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#running-steady-state

### steady

what: Converges one steady state by marching in pseudo-time, with a local time step in each cell and no physical time derivative.
when: For flows that settle to a steady state. `convergence_control` sets the scheme, CFL and number of cycles.

# GTSTimeSettings

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#global-time-stepping

### unsteady

what: Marches the solution in physical time to `total_time` in steps of `time_step`. `kind` selects dual time stepping or global time stepping.
when: For flows that vary in time. Dual time stepping converges each physical step in pseudo-time, so `time_step` is set by the time scale to be resolved. Global time stepping is explicit, so `time_step` must also be stable in the smallest cell.

## kind

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#global-time-stepping

### global timestepping

what: Advances the solution in physical time only, using the same physical time step in every cell — there is no pseudo-time convergence loop.

## scheme

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#global-time-stepping

# DTSTimeSettings

## type

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#dual-time-stepping

## kind

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#dual-time-stepping

### dual time stepping

what: Advances the solution in physical time, with an inner pseudo-time convergence loop run at each physical time step to converge that step before advancing.

## order

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#dual-time-stepping

### first

what: First-order backward difference (BDF1, implicit Euler) for the physical time derivative.
when: More dissipative in time than `second`.

### second

what: Second-order backward difference (BDF2) for the physical time derivative. Until two earlier time levels are held, the step is taken at first order.
when: The default, for time-accurate results.

### 1

what: The same as `first`.

### 2

what: The same as `second`.

## start

fingerprint: pending
guide: /guide/choosing/choosing-a-time-marching-scheme#dual-time-stepping
