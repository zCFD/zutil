# n0012_257_cL_target_1_imp.py to config version 2

update_input converted `n0012_257_cL_target_1_imp.py` (config_version 0) and wrote `n0012_257_cL_target_1_imp_v2.py`.
The new deck is a draft: 1 decision below must be made before it validates or runs.

## Files written

- `n0012_257_cL_target_1_imp_v2.py` and `n0012_257_cL_target_1_imp_v2.conversion.md`, this deck and this report

## Counts

| What happened | Count |
|---|---|
| moved | 44 |
| re-spelled | 4 |
| dropped | 4 |
| DECIDE | 1 |

## Every v1 key

| v1 key | What happened | Where it went |
|---|---|---|
| `restart` | moved | `solver > initialisation > restart` |
| `units` | dropped: the solver is always SI |  |
| `safe` | moved | `solver > solver settings > safe` |
| `reference` | moved | `solver > initialisation > initial conditions` |
| `time marching > unsteady > total time` | dropped: steady: one physical step, no time term |  |
| `time marching > unsteady > time step` | dropped: steady: one physical step, no time term |  |
| `time marching > unsteady > order` | dropped: steady: one physical step, no time term |  |
| `time marching > scheme > name` | moved | `solver > convergence control > scheme > name` |
| `time marching > scheme > stage` | moved | `solver > convergence control > scheme > stage` |
| `time marching > cfl` | moved | `solver > convergence control > cfl > cfl` |
| `time marching > cfl ramp factor > growth` | moved | `solver > convergence control > cfl > cfl ramp` |
| `time marching > cfl ramp factor > initial` | moved | `solver > convergence control > cfl > cfl ramp` |
| `time marching > cycles` | moved | `solver > convergence control > cycles` |
| `equations` | moved | `solver > equations > type` |
| `RANS > order` | moved | `solver > numerical scheme > order` |
| `RANS > limiter` | re-spelled (v2 spells the limiter 'van albada') | `solver > numerical scheme > limiter` |
| `RANS > inviscid flux scheme` | moved | `solver > numerical scheme > inviscid flux scheme` |
| `RANS > precondition` | moved | `solver > equations > precondition` |
| `RANS > turbulence > model` | moved | `solver > equations > turbulence > model` |
| `material` | re-spelled (v1 air is the ideal-gas model, which v2 names ideal gas) | `solver > fluid properties > material` |
| `air > gamma` | moved | `solver > fluid properties > gamma` |
| `air > gas constant` | moved | `solver > fluid properties > gas constant` |
| `air > Sutherlands const` | moved | `solver > fluid properties > sutherlands const` |
| `air > Prandtl No` | moved | `solver > fluid properties > prandtl no` |
| `air > Turbulent Prandtl No` | moved | `solver > fluid properties > turbulent prandtl no` |
| `IC_1 > temperature` | moved | `solver > reference conditions > IC_1 > temperature` |
| `IC_1 > pressure` | moved | `solver > reference conditions > IC_1 > pressure` |
| `IC_1 > V > vector` | moved | `solver > reference conditions > IC_1 > v > vector` |
| `IC_1 > V > Mach` | moved | `solver > reference conditions > IC_1 > v > mach` |
| `IC_1 > Reynolds No` | moved | `solver > reference conditions > IC_1 > reynolds no` |
| `IC_1 > Reference Length` | moved | `solver > reference conditions > IC_1 > reference length` |
| `IC_1 > turbulence intensity` | moved | `solver > reference conditions > IC_1 > turbulence intensity` |
| `IC_1 > eddy viscosity ratio` | moved | `solver > reference conditions > IC_1 > eddy viscosity ratio` |
| `IC_1 > ambient turbulence intensity` | moved | `solver > reference conditions > IC_1 > ambient turbulence intensity` |
| `IC_1 > ambient eddy viscosity ratio` | moved | `solver > reference conditions > IC_1 > ambient eddy viscosity ratio` |
| `IC_2` | DECIDE, item 1 | `solver > reference conditions > IC_2` |
| `BC_1 > zone` | moved | `model > naca0012_0257 > boundary conditions > BC_1 > zones` |
| `BC_1 > type` | moved | `model > naca0012_0257 > boundary conditions > BC_1 > type` |
| `BC_2 > zone` | moved | `model > naca0012_0257 > boundary conditions > BC_2 > zones` |
| `BC_2 > type` | moved | `model > naca0012_0257 > boundary conditions > BC_2 > type` |
| `BC_2 > kind` | re-spelled | `model > naca0012_0257 > boundary conditions > BC_2 > kind` |
| `BC_3 > zone` | moved | `model > naca0012_0257 > boundary conditions > BC_3 > zones` |
| `BC_3 > type` | moved | `model > naca0012_0257 > boundary conditions > BC_3 > type` |
| `BC_3 > condition` | moved | `model > naca0012_0257 > boundary conditions > BC_3 > condition` |
| `BC_3 > kind` | moved | `model > naca0012_0257 > boundary conditions > BC_3 > kind` |
| `report > frequency` | moved | `solver > output settings > report > frequency` |
| `report > forces > FR_1 > name` | moved | `solver > output settings > report > forces > FR_1 > name` |
| `report > forces > FR_1 > zone` | re-spelled | `solver > output settings > report > forces > FR_1 > zones` |
| `report > forces > FR_1 > transform` | moved (def line rewritten to the v2 call) | `solver > output settings > report > forces > FR_1 > transform` |
| `report > forces > FR_1 > reference area` | moved | `solver > output settings > report > forces > FR_1 > reference area` |
| `write output > format` | moved | `solver > output settings > solution > format` |
| `write output > surface variables` | moved | `solver > output settings > solution > surface variables` |
| `write output > volume variables` | moved | `solver > output settings > solution > volume variables` |

## Decisions

### 1. IC_2 is computed by lift_target_ic_func, and a v2 reference condition is a fixed state. Which state should IC_2 hold?

v2 key: `solver > reference conditions > IC_2`

Write IC_2 as a reference condition with the vector the run starts from. To trim alpha to a lift coefficient, give BC_3 a target, such as "target": {"alpha": {"cl": ...}}; the solver turns alpha until cl is met, then holds it. lift_target_ic_func and the code it uses no longer run: delete them. A force-report transform that rotated by its angle stays at the starting angle; the target computes cl itself.

v1 keys: `IC_2` = `lift_target_ic_func`

## Notes

- my_transform: `def my_transform(x, y, z):` is now `def my_transform(*, vector):`; its body still returns the v1/v2/v3 dict, which v2 accepts (`report > forces > FR_1 > transform`)
