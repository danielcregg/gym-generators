from setuptools import setup, find_packages

setup(
    name='gym_generators',
    version='1.0.0',
    keywords='environment, agent, rl, gymnasium, reinforcement-learning, ceed',
    url='https://github.com/danielcregg/gym-generators',
    description='Combined Economic and Emission Dispatch (CEED) environment for Gymnasium',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    entry_points={
        'gymnasium.envs': ['__root__ = gym_generators:__init__'],
    },
    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
    ],
    extras_require={
        'train': [
            'stable-baselines3>=2.0.0',
            'matplotlib>=3.5.0',
        ],
        'test': [
            'pytest>=7.0.0',
        ],
    },
    python_requires='>=3.8',
)
