from distutils.core import setup

setup(name="zutil",
      version='0.1.5',
      description="Utilities used for generating zCFD control dictionaries",
      author="Zenotech",
      author_email="support@zenotech.com",
      url="https://zcfd.zenotech.com/",
      packages=["zutil", "zutil.post", "zutil.analysis", "zutil.plot"],
      install_requires=[
          'ipython<6.0',
          'Fabric',
          'ipywidgets',
          'matplotlib',
          'numpy',
          'pandas',
          'PyYAML'
      ],
      extras_require={
          "mpi": ["mpi4py"]
      }
      )
