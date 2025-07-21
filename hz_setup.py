# pip install setuptools wheel
# python hz_setup.py sdist bdist_wheel
from setuptools import setup, find_packages

setup(
    name='hipporag',
    version='0.1',
    package_dir={"": "src"},
    packages=find_packages("src"),
    py_modules=["hz_graphrag"],  # 显式包含单独的模块
    install_requires=[  # 项目依赖
        "python_igraph==0.11.8",
    ],
    python_requires=">=3.10",
    # 其他元数据
    author='Haizhi',
    author_email='admin@haizhi.com',
    description='Haizhi GraphRag',
    entry_points={  # 注册命令行工具
        'console_scripts': [
            'hz_graphrag=hz_graphrag:main',  # 这里定义了命令和对应的函数
        ],
    },
)
