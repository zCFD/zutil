# zutil

[![PyPI version fury.io](https://badge.fury.io/py/zutil.svg)](https://pypi.python.org/pypi/zutil/)

**Python utilities for zCFD users**

zutil is a comprehensive Python toolkit designed to work with zCFD CFD simulation results and provide powerful pre- and post-processing capabilities. This library offers streamlined workflows for data analysis, visualization, wind energy modeling, acoustic analysis, and geometric transformations.

## Installation

Install zutil from PyPI:

```bash
pip install zutil
```

**Note:** This library is compatible with Paraview 6.0+

## Package Structure

zutil is organized into several specialized modules:

- **[plot](#plotting)** - Advanced plotting and visualization tools for zCFD results
- **[post](#post-processing)** - Post-processing utilities using Paraview integration
- **[fileutils](#file-utilities)** - File I/O operations and result data access
- **[analysis](#analysis-tools)** - Analysis tools including acoustic analysis
- **[transform](#geometric-transformations)** - Geometric transformation utilities
- **[zWind](#wind-energy-modeling)** - Wind energy modeling and turbine analysis

## Quick Start

### Loading zCFD Results

The fundamental function for accessing zCFD simulation data:

```python
from zutil.fileutils import get_zcfd_result

# Load a zCFD case result
result = get_zcfd_result('my_case.py')

# Access case parameters
params = result.parameters

# Get report data
report = result.report
print(report.header_list)  # Available data columns
```

### Plotting Report Files

Create professional plots from zCFD report files:

```python
from zutil.plot import Report
import matplotlib.pyplot as plt

# Load and plot residuals
report = Report('my_case.py')
report.plot_residuals()
plt.show()

# Plot specific variables
report.plot_report_array('Cd')
plt.show()
```

## Module Documentation

### Plotting

The `zutil.plot` module provides comprehensive plotting capabilities:

```python
from zutil.plot import Report, zCFD_Plots

# Individual case plotting
report = Report('case.py')
report.plot_residuals()
report.plot_forces()
report.plot_linear_solve_performance()

# Multi-case plotting (handles overset cases automatically)
plots = zCFD_Plots('case.py')
plots.plot_residuals()
plots.plot_forces()
```

**Key Features:**

- Residual convergence plots
- Force and moment coefficient plotting
- Linear solver performance analysis
- Acoustic analysis visualization
- Professional styling with zCFD branding

### Post-Processing

The `zutil.post` module integrates with Paraview for advanced post-processing:

**Note:** This module requires `pvpython` to be installed. The easiest way to get this is to use a distribution of zCFD.

```python
from zutil import post

# Access case parameters
params = post.get_case_parameters('case.py')

# Calculate forces on walls
force = post.calc_force_wall('wall_name', 'case.py')

# Extract surface pressure profiles
cp_data = post.cp_profile_wall_from_file('wall_name', 'case.py')

# Generate Paraview visualizations with zCFD branding
post.vtk_logo_stamp()
```

**Key Features:**

- Force and moment calculations
- Surface pressure and friction analysis
- Paraview visualization utilities
- Progress tracking for batch operations

### File Utilities

The `zutil.fileutils` module handles all file I/O operations:

```python
from zutil.fileutils import get_zcfd_result, zCFD_Result

# Load case data
result = get_zcfd_result('case.py')

# Access report data directly
report_data = result.report.get_array('Cd')

# Get case status
success = result.report.get_case_success()
```

**Key Features:**

- Seamless access to zCFD case data
- Report file parsing and analysis
- Parameter extraction from control files
- CSV data handling utilities

### Analysis Tools

The `zutil.analysis` module includes acoustic and other analysis capabilities:

```python
from zutil.analysis.acoustic import calculate_PSD
from zutil.plot import plot_PSD, plot_thirdoctave

# Perform acoustic analysis
psd_data = calculate_PSD(time_series_data, sample_rate)

# Create acoustic plots
plot_PSD(psd_data, frequencies)
plot_thirdoctave(psd_data, frequencies)
```

**Key Features:**

- Power spectral density (PSD) analysis
- Third-octave band analysis
- Acoustic visualization tools
- Signal processing utilities

### Geometric Transformations

The `zutil.transform` module provides coordinate and geometric operations:

```python
from zutil.transform import transform
from zutil import vector_from_angle, rotate_vector

# Coordinate transformations
transformed_points = transform.apply_rotation(points, rotation_matrix)

# Vector operations
wind_vector = vector_from_angle(wind_direction, wind_speed)
rotated_vector = rotate_vector(vector, axis, angle)
```

**Key Features:**

- Coordinate system transformations
- Vector mathematics utilities
- Rotation and translation operations
- Unit conversions

### Wind Energy Modeling

The `zutil.zWind` module provides specialized wind energy tools:

```python
from zutil.zWind import zTurbine

# Define wind turbine properties
turbine = zTurbine(name='NREL_5MW', centre=[0, 0, 90])
```

**Key Features:**

- Wind turbine modeling (zTurbine)
- Actuator disc implementations (zActuatorDisc)
- Blade element momentum theory (zBem)
- Turbine controllers (zController)

## Requirements

- Python 3.9+
- NumPy
- Pandas
- Matplotlib
- IPython/Jupyter support
- PyYAML
- Paraview (for post-processing features)

## Documentation

For comprehensive documentation, tutorials, and examples:

- **Full Documentation:** [https://docs.zenotech.com/](https://docs.zenotech.com/)
- **zCFD Solver:** Download a free trial at [https://zcfd.zenotech.com](https://zcfd.zenotech.com)

## Support

- **Issues:** Report bugs and feature requests via the [Issues tab](https://github.com/zenotech/zCFD/issues)
- **Email:** For technical support contact [admin@zenotech.com](mailto:admin@zenotech.com)

## License

zutil is released under the MIT License. See LICENSE file for details.

---

_Developed by [Zenotech Ltd](https://zenotech.com)_
