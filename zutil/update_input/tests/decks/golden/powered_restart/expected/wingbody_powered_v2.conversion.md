# wingbody_powered.py to config version 2

update_input converted `wingbody_powered.py` (config_version 0) and wrote `wingbody_powered_v2.py`.
The new deck passed config version 2 validation.

## Files written

- `wingbody_powered_v2.py` and `wingbody_powered_v2.conversion.md`, this deck and this report

## Counts

| What happened | Count |
|---|---|
| moved | 94 |
| re-spelled | 24 |
| dropped | 11 |
| DECIDE | 0 |

## Every v1 key

| v1 key | What happened | Where it went |
|---|---|---|
| `scale` | moved | `model > wingbody > transforms > scale` |
| `units` | dropped: the solver is always SI |  |
| `reference` | moved | `solver > initialisation > reference` |
| `initial > name` | moved | `solver > initialisation > initial conditions` |
| `restart` | moved | `solver > initialisation > restart` |
| `restart casename` | moved | `solver > initialisation > restart casename` |
| `restart ignore history` | moved | `solver > initialisation > restart ignore history` |
| `safe` | moved | `solver > solver settings > safe` |
| `partitioner` | dropped: metis is a development partitioner; scotch is the default |  |
| `time marching > unsteady > total time` | dropped: steady: one physical step, no time term |  |
| `time marching > unsteady > time step` | dropped: steady: one physical step, no time term |  |
| `time marching > unsteady > order` | dropped: steady: one physical step, no time term |  |
| `time marching > unsteady > start` | dropped: steady: one physical step, no time term |  |
| `time marching > scheme > name` | moved | `solver > convergence control > scheme > name` |
| `time marching > scheme > stage` | moved | `solver > convergence control > scheme > stage` |
| `time marching > cfl` | moved | `solver > convergence control > cfl > cfl` |
| `time marching > cfl transport` | moved | `solver > convergence control > cfl > cfl turbulence` |
| `time marching > cfl ramp factor > initial` | moved | `solver > convergence control > cfl > cfl ramp` |
| `time marching > cfl ramp factor > growth` | moved | `solver > convergence control > cfl > cfl ramp` |
| `time marching > cycles` | moved | `solver > convergence control > cycles` |
| `linear solver options > double precision` | moved | `solver > solver settings > linear solver options > double precision` |
| `equations` | moved | `solver > equations > type` |
| `rans > order` | moved | `solver > numerical scheme > order` |
| `rans > limiter` | re-spelled (v2 spells the limiter 'van albada') | `solver > numerical scheme > limiter` |
| `rans > precondition` | moved | `solver > equations > precondition` |
| `rans > inviscid flux scheme` | moved | `solver > numerical scheme > inviscid flux scheme` |
| `rans > first order cycles` | moved | `solver > numerical scheme > first order cycles` |
| `rans > turbulence > model` | moved | `solver > equations > turbulence > model` |
| `rans > turbulence > qcr` | moved | `solver > equations > turbulence > qcr` |
| `material` | re-spelled (v1 air is the ideal-gas model, which v2 names ideal gas) | `solver > fluid properties > material` |
| `air > gamma` | moved | `solver > fluid properties > gamma` |
| `air > gas constant` | moved | `solver > fluid properties > gas constant` |
| `air > Sutherlands const` | moved | `solver > fluid properties > sutherlands const` |
| `air > Prandtl No` | moved | `solver > fluid properties > prandtl no` |
| `air > Turbulent Prandtl No` | moved | `solver > fluid properties > turbulent prandtl no` |
| `IC_1 > temperature` | moved | `solver > reference conditions > IC_1 > temperature` |
| `IC_1 > pressure` | moved | `solver > reference conditions > IC_1 > pressure` |
| `IC_1 > V > Mach` | moved | `solver > reference conditions > IC_1 > v > mach` |
| `IC_1 > V > vector` | moved | `solver > reference conditions > IC_1 > v > vector` |
| `IC_1 > viscosity` | moved | `solver > reference conditions > IC_1 > viscosity` |
| `IC_1 > turbulence intensity` | moved | `solver > reference conditions > IC_1 > turbulence intensity` |
| `IC_1 > eddy viscosity ratio` | moved | `solver > reference conditions > IC_1 > eddy viscosity ratio` |
| `IC_1 > ambient turbulence intensity` | moved | `solver > reference conditions > IC_1 > ambient turbulence intensity` |
| `IC_1 > ambient eddy viscosity ratio` | moved | `solver > reference conditions > IC_1 > ambient eddy viscosity ratio` |
| `IC_2 > reference` | dropped: inlined into BC_4, whose reference is IC_1 |  |
| `IC_2 > mass flow rate` | moved | `model > wingbody > boundary conditions > BC_4 > mass flow rate` |
| `IC_3 > reference` | dropped: inlined into BC_5, whose reference is IC_1 |  |
| `IC_3 > mass flow rate` | moved | `model > wingbody > boundary conditions > BC_5 > mass flow rate` |
| `IC_3 > total temperature ratio` | moved | `model > wingbody > boundary conditions > BC_5 > total temperature ratio` |
| `IC_4 > reference` | dropped: inlined into BC_6, whose reference is IC_1 |  |
| `IC_4 > mass flow rate` | moved | `model > wingbody > boundary conditions > BC_6 > mass flow rate` |
| `IC_5 > reference` | dropped: inlined into BC_7, whose reference is IC_1 |  |
| `IC_5 > total pressure ratio` | moved | `model > wingbody > boundary conditions > BC_7 > total pressure ratio` |
| `IC_5 > total temperature ratio` | moved | `model > wingbody > boundary conditions > BC_7 > total temperature ratio` |
| `IC_6 > reference` | dropped: inlined into BC_8, whose reference is IC_1 |  |
| `IC_6 > total pressure ratio` | moved | `model > wingbody > boundary conditions > BC_8 > total pressure ratio` |
| `IC_6 > total temperature ratio` | moved | `model > wingbody > boundary conditions > BC_8 > total temperature ratio` |
| `BC_1 > zone` | moved | `model > wingbody > boundary conditions > BC_1 > zones` |
| `BC_1 > type` | moved | `model > wingbody > boundary conditions > BC_1 > type` |
| `BC_2 > zone` | moved | `model > wingbody > boundary conditions > BC_2 > zones` |
| `BC_2 > type` | moved | `model > wingbody > boundary conditions > BC_2 > type` |
| `BC_2 > kind` | re-spelled | `model > wingbody > boundary conditions > BC_2 > kind` |
| `BC_3 > zone` | moved | `model > wingbody > boundary conditions > BC_3 > zones` |
| `BC_3 > type` | moved | `model > wingbody > boundary conditions > BC_3 > type` |
| `BC_3 > condition` | moved | `model > wingbody > boundary conditions > BC_3 > condition` |
| `BC_3 > kind` | moved | `model > wingbody > boundary conditions > BC_3 > kind` |
| `BC_4 > zone` | moved | `model > wingbody > boundary conditions > BC_4 > zones` |
| `BC_4 > type` | moved | `model > wingbody > boundary conditions > BC_4 > type` |
| `BC_4 > kind` | moved | `model > wingbody > boundary conditions > BC_4 > kind` |
| `BC_4 > condition` | re-spelled (IC_2 is derived from IC_1; its ratios go on BC_4) | `model > wingbody > boundary conditions > BC_4 > reference` |
| `BC_5 > zone` | moved | `model > wingbody > boundary conditions > BC_5 > zones` |
| `BC_5 > type` | moved | `model > wingbody > boundary conditions > BC_5 > type` |
| `BC_5 > kind` | moved | `model > wingbody > boundary conditions > BC_5 > kind` |
| `BC_5 > condition` | re-spelled (IC_3 is derived from IC_1; its ratios go on BC_5) | `model > wingbody > boundary conditions > BC_5 > reference` |
| `BC_6 > zone` | moved | `model > wingbody > boundary conditions > BC_6 > zones` |
| `BC_6 > type` | moved | `model > wingbody > boundary conditions > BC_6 > type` |
| `BC_6 > kind` | moved | `model > wingbody > boundary conditions > BC_6 > kind` |
| `BC_6 > condition` | re-spelled (IC_4 is derived from IC_1; its ratios go on BC_6) | `model > wingbody > boundary conditions > BC_6 > reference` |
| `BC_7 > zone` | moved | `model > wingbody > boundary conditions > BC_7 > zones` |
| `BC_7 > type` | moved | `model > wingbody > boundary conditions > BC_7 > type` |
| `BC_7 > kind` | re-spelled (v1 treated every compressible inflow that was not massflow as pressure) | `model > wingbody > boundary conditions > BC_7 > kind` |
| `BC_7 > condition` | re-spelled (IC_5 is derived from IC_1; its ratios go on BC_7) | `model > wingbody > boundary conditions > BC_7 > reference` |
| `BC_8 > zone` | moved | `model > wingbody > boundary conditions > BC_8 > zones` |
| `BC_8 > type` | moved | `model > wingbody > boundary conditions > BC_8 > type` |
| `BC_8 > kind` | re-spelled (v1 treated every compressible inflow that was not massflow as pressure) | `model > wingbody > boundary conditions > BC_8 > kind` |
| `BC_8 > condition` | re-spelled (IC_6 is derived from IC_1; its ratios go on BC_8) | `model > wingbody > boundary conditions > BC_8 > reference` |
| `write output > format` | moved | `solver > output settings > solution > format` |
| `write output > surface variables` | moved | `solver > output settings > solution > surface variables` |
| `write output > volume variables` | moved | `solver > output settings > solution > volume variables` |
| `write output > frequency` | moved | `solver > output settings > solution > frequency` |
| `report > frequency` | moved | `solver > output settings > report > frequency` |
| `report > forces > FR_1 > name` | moved | `solver > output settings > report > forces > FR_1 > name` |
| `report > forces > FR_1 > zone` | re-spelled | `solver > output settings > report > forces > FR_1 > zones` |
| `report > forces > FR_1 > transform` | moved (def line rewritten to the v2 call) | `solver > output settings > report > forces > FR_1 > transform` |
| `report > forces > FR_1 > reference area` | moved | `solver > output settings > report > forces > FR_1 > reference area` |
| `report > forces > FR_1 > reference point` | moved | `solver > output settings > report > forces > FR_1 > reference point` |
| `report > forces > FR_1 > reference length` | moved | `solver > output settings > report > forces > FR_1 > reference length` |
| `report > forces > FR_1 > reference pressure` | moved | `solver > output settings > report > forces > FR_1 > reference pressure` |
| `report > forces > FR_2 > name` | moved | `solver > output settings > report > forces > FR_2 > name` |
| `report > forces > FR_2 > zone` | re-spelled | `solver > output settings > report > forces > FR_2 > zones` |
| `report > forces > FR_2 > transform` | moved (def line rewritten to the v2 call) | `solver > output settings > report > forces > FR_2 > transform` |
| `report > forces > FR_2 > reference area` | moved | `solver > output settings > report > forces > FR_2 > reference area` |
| `report > forces > FR_2 > reference point` | moved | `solver > output settings > report > forces > FR_2 > reference point` |
| `report > forces > FR_2 > reference length` | moved | `solver > output settings > report > forces > FR_2 > reference length` |
| `report > forces > FR_2 > reference pressure` | moved | `solver > output settings > report > forces > FR_2 > reference pressure` |
| `report > forces > FR_3 > name` | moved | `solver > output settings > report > forces > FR_3 > name` |
| `report > forces > FR_3 > zone` | re-spelled | `solver > output settings > report > forces > FR_3 > zones` |
| `report > forces > FR_3 > transform` | moved (def line rewritten to the v2 call) | `solver > output settings > report > forces > FR_3 > transform` |
| `report > forces > FR_3 > reference area` | moved | `solver > output settings > report > forces > FR_3 > reference area` |
| `report > forces > FR_3 > reference point` | moved | `solver > output settings > report > forces > FR_3 > reference point` |
| `report > forces > FR_3 > reference length` | moved | `solver > output settings > report > forces > FR_3 > reference length` |
| `report > forces > FR_3 > reference pressure` | moved | `solver > output settings > report > forces > FR_3 > reference pressure` |
| `report > forces > FR_4 > name` | moved | `solver > output settings > report > forces > FR_4 > name` |
| `report > forces > FR_4 > zone` | re-spelled | `solver > output settings > report > forces > FR_4 > zones` |
| `report > forces > FR_4 > transform` | moved (def line rewritten to the v2 call) | `solver > output settings > report > forces > FR_4 > transform` |
| `report > forces > FR_4 > reference area` | moved | `solver > output settings > report > forces > FR_4 > reference area` |
| `report > forces > FR_4 > reference point` | moved | `solver > output settings > report > forces > FR_4 > reference point` |
| `report > forces > FR_4 > reference length` | moved | `solver > output settings > report > forces > FR_4 > reference length` |
| `report > forces > FR_4 > reference pressure` | moved | `solver > output settings > report > forces > FR_4 > reference pressure` |
| `report > massflow > MF_1 > name` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > intake_duct > zones` |
| `report > massflow > MF_1 > zone` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > intake_duct > zones` |
| `report > massflow > MF_2 > name` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > exhaust_duct > zones` |
| `report > massflow > MF_2 > zone` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > exhaust_duct > zones` |
| `report > massflow > MF_3 > name` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > fan_face > zones` |
| `report > massflow > MF_3 > zone` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > fan_face > zones` |
| `report > massflow > MF_4 > name` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > cold_nozzle > zones` |
| `report > massflow > MF_4 > zone` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > cold_nozzle > zones` |
| `report > massflow > MF_5 > name` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > hot_nozzle > zones` |
| `report > massflow > MF_5 > zone` | re-spelled (the report name becomes the entry's key) | `solver > output settings > report > mass flow > hot_nozzle > zones` |

## Notes

- The legacy validator cannot read a file the deck names from here; check it is in place before the run: restart casename: File does not exist
- partitioner 'metis' is a development setting, available only in research versions of zCFD: the converted deck partitions with scotch, the default. (`partitioner`)
- body_axes: `def body_axes(x, y, z):` is now `def body_axes(*, vector):`; its body still returns the v1/v2/v3 dict, which v2 accepts (`report > forces > FR_1 > transform`, `report > forces > FR_2 > transform`, `report > forces > FR_3 > transform`, `report > forces > FR_4 > transform`)
