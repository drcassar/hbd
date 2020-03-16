import setuptools

setuptools.setup(
    name='hbd',
    version='1.6',
    author='Daniel Roberto Cassar',
    author_email='daniel.r.cassar@gmail.com',
    description='HBD',
    url="https://github.com/drcassar/hbd",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.1', 'scipy>=0.19', 'pandas>=0.24.0', 'lmfit>=0.9.13', 'deap',
        'tensorflow'
    ],
    python_requires='>=3.6',
)
