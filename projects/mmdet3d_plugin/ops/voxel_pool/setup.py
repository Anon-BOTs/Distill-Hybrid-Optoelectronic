from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='voxel_pool',
    ext_modules=[
        CppExtension('voxel_pool_ext', [
            'src/voxel_pool.cpp',
            'src/voxel_pool_cuda.cu'
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
