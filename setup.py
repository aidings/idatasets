from setuptools import find_packages, setup

setup(name='idatasets',  # 包名
      version='1.0.0',  # 版本号
      description='',
      long_description='',
      author='zhifeng.ding',
      author_email='ioimage@163.com',
      url='https://github.com/aidings/idataset.git',
      license='',
      install_requires=['torchvision', 'numpy', 'opencv-python', 'pillow', 'scipy', 'scikit-image', 'pandas', 'tqdm'],
      extras_require={},
      dependency_links=[
          "https://pypi.tuna.tsinghua.edu.cn/simple",
          "http://mirrors.aliyun.com/pypi/simple"
      ],
      classifiers=[
          'Intended Audience :: Developers',
          'Operating System :: OS Independent',
          'Natural Language :: Chinese (Simplified)',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.2'
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Topic :: Utilities'
      ],
      keywords='',
      packages=find_packages('src', exclude=["examples", "tests", "project"]),  # 必填
      package_dir={'': 'src'},  # 必填
      include_package_data=True,
      scripts= [
      ],
)