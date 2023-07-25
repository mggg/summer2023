import setuptools

setuptools.setup(
    name='sklearn_pyclustering_shim',
    version='0.1.0',    
    description='A small shim to use the PyClustering XMeans implementation with Sklearn API',
    author='Hazel Brenner',
    author_email='hnb29@cornell.edu',
    license='BSD 2-clause',
    packages=setuptools.find_packages(),
    install_requires=['sklearn',
                      'pyclustering',
                      'numpy',                     
                      ],
)