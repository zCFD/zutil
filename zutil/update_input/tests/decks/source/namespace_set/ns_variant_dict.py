from zutil import get_parameters_from_file

parameters = get_parameters_from_file(
    "ns_base.py", namespace={"mach": 0.3, "t_end": 2.0, "cycles": 5}
)
