from distutils.core import setup

setup(name="zutil",
      version='0.2.0',
      description="Utilities used for generating zCFD control dictionaries",
      author="Zenotech",
      author_email="support@zenotech.com",
      license="MIT",
      url="https://zcfd.zenotech.com/",
      project_urls={
          "Source Code": "https://github.com/zCFD/zutil/",
      },
      packages=["zutil", "zutil.post", "zutil.analysis", "zutil.plot"],
      install_requires=[
          'future',
          'ipython',
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
