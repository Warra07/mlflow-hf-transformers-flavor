from setuptools import setup

setup(name='mlflow_hf_transformers',
      version='0.3',
      description='Mlflow flavors for pytorch huggingface transformers models',
          install_requires=[
        "mlflow>=1.11.0",
        "transformers[torch]>=4.10.3",
    ],
      keywords='mlflow huggingface transformers',
      url='https://github.com/Warra07/mlflow-hf-transformers-flavor',
      author='Wacim Belahcel',
      author_email='wacimbelahcel@gmail.com',
      license='Apache 2',
      packages=['mlflow_hf_transformers'])