from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='smooth_sampler_ext',
    ext_modules=[
        CppExtension('smooth_sampler_ext', [
            'src/smooth_sampler.cpp',
            'src/smooth_sampler_cuda.cu'
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
