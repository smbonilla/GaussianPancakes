from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name='gaussian-norms',
    version='0.1.0',
    author='Sierra Bonilla',
    author_email='sierra.bonilla.21@ucl.ac.uk',
    description='Custom CUDA extension for computing Gaussian normals.',
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='gaussian_norms._C',
            sources=[
                'csrc/forward.cu',
                'csrc/backward.cu',
                'csrc/computeGaussNorm.cu',
                'csrc/ext.cpp'
            ],
            include_dirs=[],  
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]},  
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    license='LICENSE.md',
    keywords='gaussian norms cuda pytorch extension',
    install_requires=[
        'torch',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)