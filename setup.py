from distutils.core import setup

setup(name="zutil",
      version='0.1.2',
      description="Utilities used for generating zCFD control dictionaries",
      author="Zenotech",
      author_email="support@zenotech.com",
      url="https://zcfd.zenotech.com/",
      packages=["zutil", "zutil.post", "zutil.analysis", "zutil.plot"],
      install_requires=[
          'mpi4py',
          'ipython',
          'Fabric',
          'ipywidgets',
          'matplotlib',
          'numpy',
          'pandas',
          'PyYAML'
        ],
      )