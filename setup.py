from distutils.core import setup

classes = """
    Development Status :: 4 - Beta
    Intended Audience :: Developers
    License :: OSI Approved :: MIT License
    Topic :: System :: Logging
    Programming Language :: Python
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.7
    Operating System :: POSIX :: Linux
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

setup(name="zutil",
      version='1.0.2',
      description="Utilities used for generating zCFD control dictionaries",
      author="Zenotech",
      author_email="support@zenotech.com",
      license="MIT",
      url="https://zcfd.zenotech.com/",
      classifiers=classifiers,
      project_urls={
          "Source Code": "https://github.com/zCFD/zutil/",
      },
      packages=["zutil", "zutil.post", "zutil.analysis", "zutil.plot"],
      install_requires=[
          'future',
          'ipython',
          'fabric>=2.5',
          'ipywidgets',
          'matplotlib',
          'numpy',
          'pandas',
          'PyYAML',
	  'dill'
      ],
      extras_require={
          "mpi": ["mpi4py"]
      }
      )
