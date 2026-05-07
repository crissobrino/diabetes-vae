"""Setup script for diabetes-vae package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / 'README.md'
long_description = readme_path.read_text(encoding='utf-8') if readme_path.exists() else ''

# Read requirements
requirements_path = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='diabetes-vae',
    version='0.1.0',
    description='Latent Metabolic State Learning with Variational Autoencoders',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='[Your Name]',
    author_email='[Your Email]',
    url='https://github.com/[username]/diabetes-vae',
    license='[License Type]',
    
    packages=find_packages(),
    
    install_requires=requirements,
    
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.9.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
    },
    
    python_requires='>=3.8',
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: [License]',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
    
    keywords='vae variational-autoencoder diabetes cgm glucose metabolic-state',
    
    project_urls={
        'Bug Reports': 'https://github.com/[username]/diabetes-vae/issues',
        'Source': 'https://github.com/[username]/diabetes-vae',
    },
)
