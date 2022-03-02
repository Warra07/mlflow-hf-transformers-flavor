# Mlflow huggingface transformer flavor

### Introduction

A simple flavor to save and load huggingface transformers model on MLflow

### Usage



``` Python
import mlflow_hf_transformers
import mlflow
import transformers
with mlflow.start_run() as run: 
  mlflow_hf_transformers.log_model(model=model,artifact_path="testmodel", tokenizer=tokenizer)
 
 
import mlflow_hf_transformers

import mlflow
logged_model = 'runs:/xxxxxx/testmodel'

loaded_model, tokenizer = mlflow_hf_transformers.load_model(logged_model)
```

Tokenizer is optional, you can also save and load the tokenizer as an artifact :



``` Python
import mlflow_hf_transformers
import mlflow
import transformers
with mlflow.start_run() as run: 
  mlflow_hf_transformers.log_tokenizer(tokenizer,"tokenizer")

  
logged_model = 'runs:/xxxx/tokenizer'

loaded_tokenizer = mlflow_hf_transformers.load_tokenizer(logged_model)

```



### Installation

1. Make sure pip is installed (https://packaging.python.org/tutorials/installing-packages/)<br/>
2. Then you can install the flavor:<br/>
`> pip install git+https://github.com/Warra07/mlflow-hf-transformers-flavor.git`
