from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

packages = [req.split('==')[0] for req in requirements]

setup(
  name='Indox',
  version='0.1.1',
  license='MIT',
  packages=find_packages(),
  package_data={'Indox': ['config.yaml']},
  include_package_data=True,
  description='Indox Retrieval Augmentation',
  author='nerdstudio',
  author_email='Mohammad@nematifamilyfundation.onmicrosoft.com',
  url='https://github.com/osllmai/inDox',
  keywords=['RAG', 'LLM'],
  install_requires=packages, 
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
