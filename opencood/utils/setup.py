from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("opencood.utils.cython.box_utils", ["opencood/utils/cython/box_utils.pyx"]),
    Extension("opencood.utils.cython.pcd_utils", ["opencood/utils/cython/pcd_utils.pyx"]),
    Extension("opencood.utils.cython.transformation_utils", ["opencood/utils/cython/transformation_utils.pyx"]),
    Extension("opencood.data_utils.post_processor.voxel_cython", ["opencood/data_utils/post_processor/voxel_cython.pyx"]),
]
# # 如果需要包含其他头文件，可以在这里添加
#         # include_dirs=[numpy.get_include()], language="c",
# 命令 (在 opencood 里面执行) `python opencood/utils/setup.py build_ext --inplace`
setup(
    name="opencood",
    version="0.1",
    packages=[
        "opencood", "opencood.utils", "opencood.utils.cython", "opencood.data_utils", "opencood.data_utils.post_processor" # fmt: skip
    ],
    ext_modules=cythonize(
        extensions,
        language_level=3,
        build_dir="build/cython",  # 指定生成 .c 文件的目录
        nthreads=4,  # 根据需要设置并行编译线程数
    ),
    include_dirs=[numpy.get_include()],
    zip_safe=False,
    # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], # 给每个 Extension 里面添加, 实在是比较麻烦
)
