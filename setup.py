from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fashion-clip',
    version='0.0.1',
    description='',
    author='Jacopo Tagliabue, Patrick John Chia, Federico Bianchi',
    author_email='jtagliabue@coveo.com, pchia@coveo.com, f.bianchi@unibocconi.it',
    packages=[
        'fashion_clip',
    ],
    classifiers=[
    ],
    install_requires=requirements,
    zip_safe=False
    )