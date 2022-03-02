"""
The `mlflow_vismod` module provides an API for logging and loading Vega models. This module
exports Vega models with the following flavors:

Vega (native) format
    This is the main flavor that can be loaded back into Vega.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
# Standard Libraries
import logging
import os
import yaml
import warnings


from packaging.version import Version
import posixpath

import mlflow
import shutil
import mlflow.pyfunc.utils as pyfunc_utils
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.file_utils import _copy_file_or_tree, TempDir, write_to
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

# Internal Libraries
import mlflow_hf_transformers


FLAVOR_NAME = "mlflow_hf_transformers"
_MODEL_DIR_SUBPATH = "model"
_TOKENIZER_DIR_SUBPATH = "tokenizer"
_EXTRA_FILES_KEY = "extra_files"
_REQUIREMENTS_FILE_KEY = "requirements_file"



_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return list(
        map(
            _get_pinned_requirement,
            [
                "torch",
                "transformers",
                "cloudpickle",
            ],
        )
    )


def get_default_conda_env():
    """
    :return: The default Conda environment as a dictionary for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.

    .. code-block:: python
        :caption: Example

        import mlflow.pytorch

        # Log PyTorch model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

        # Fetch the associated conda environment
        env = mlflow.pytorch.get_default_conda_env()
        print("conda env: {}".format(env))

    .. code-block:: text
        :caption: Output

        conda env {'name': 'mlflow-env',
                   'channels': ['conda-forge'],
                   'dependencies': ['python=3.7.5',
                                    {'pip': ['torch==1.5.1',
                                             'mlflow',
                                             'cloudpickle==1.6.0']}]}
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())



def log_model(
    model,
    artifact_path,
    tokenizer=None,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log a transformers model as an MLflow artifact for the current run.



    :param model: transformers model to be saved. 

    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.

        :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.

    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    :param requirements_file:

        .. warning::

            ``requirements_file`` has been deprecated. Please use ``pip_requirements`` instead.

        A string containing the path to requirements file. Remote URIs are resolved to absolute
        filesystem paths. For example, consider the following ``requirements_file`` string:


    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -

                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]

                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

                      If ``None``, no extra files are added to the model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.


    """
    import torch, transformers

    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_hf_transformers,
        model=model,
        tokenizer=tokenizer,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        extra_files=extra_files,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )



def save_model(
    model,
    path,
    tokenizer=None,
    conda_env=None,
    mlflow_model=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """Save a transformers model to a local file or a run with tokenizer (optional).


    """
    import torch, transformers

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    if not isinstance(model, transformers.PreTrainedModel):
        raise TypeError("Argument 'model' should be a transformers.PreTrainedModel")

    
    if code_paths is not None:
        if not isinstance(code_paths, list):
            raise TypeError("Argument code_paths should be a list, not {}".format(type(code_paths)))

    path = os.path.abspath(path)


    if mlflow_model is None:
        mlflow_model = Model()
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)



    model_data_subpath = "data"
    model_data_path = os.path.join(path, model_data_subpath)
    if not os.path.exists(model_data_path):
        os.makedirs(model_data_path)

    model_path = os.path.join(model_data_path, _MODEL_DIR_SUBPATH)
    
    
    model.save_pretrained(model_path)

    if tokenizer is not None:
        tokenizer_path = os.path.join(model_data_path, _TOKENIZER_DIR_SUBPATH)
        tokenizer.save_pretrained(tokenizer_path)


    torchserve_artifacts_config = {}

    if extra_files:
        torchserve_artifacts_config[_EXTRA_FILES_KEY] = []
        if not isinstance(extra_files, list):
            raise TypeError("Extra files argument should be a list")

        with TempDir() as tmp_extra_files_dir:
            for extra_file in extra_files:
                _download_artifact_from_uri(
                    artifact_uri=extra_file, output_path=tmp_extra_files_dir.path()
                )
                rel_path = posixpath.join(_EXTRA_FILES_KEY, os.path.basename(extra_file))
                torchserve_artifacts_config[_EXTRA_FILES_KEY].append({"path": rel_path})
            shutil.move(
                tmp_extra_files_dir.path(),
                posixpath.join(path, _EXTRA_FILES_KEY),
            )


    if code_paths is not None:
        code_dir_subpath = "code"
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir=code_dir_subpath)
    else:
        code_dir_subpath = None



    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_data=model_data_subpath,
        pytorch_version=str(torch.__version__),
        transformers_version=str(transformers.__version__),
        model_class_name=model.__class__.__name__,  
        **torchserve_artifacts_config,
    )
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow_hf_transformers",
        data=model_data_subpath,
        code=code_dir_subpath,
        env=_CONDA_ENV_FILE_NAME,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    conda_env_subpath = "conda.yaml"

    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)
    # Save `constraints.txt` if necessary



    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))




def load_model(model_uri, dst_path=None):
    """
    Load a transformers model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A transformers model.


    """
    import torch, transformers



    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    try:
        pyfunc_conf = _get_flavor_configuration(
            model_path=local_model_path, flavor_name=pyfunc.FLAVOR_NAME
        )
    except MlflowException:
        pyfunc_conf = {}
    code_subpath = pyfunc_conf.get(pyfunc.CODE)
    if code_subpath is not None:
        pyfunc_utils._add_code_to_system_path(
            code_path=os.path.join(local_model_path, code_subpath)
        )

    hf_transformers_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)

    if torch.__version__ != hf_transformers_conf["pytorch_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed PyTorch version '%s'",
            hf_transformers_conf["pytorch_version"],
            torch.__version__,
        ) 
    if transformers.__version__ != hf_transformers_conf["transformers_version"]:
        _logger.warning(
            "Stored model version '%s' does not match installed transformers version '%s'",
            hf_transformers_conf["transformers_version"],
            transformers.__version__,
        )     

    transformers_model_artifacts_path = os.path.join(local_model_path, hf_transformers_conf["model_data"])
    return _load_model(path=transformers_model_artifacts_path,  
    model_class_name=hf_transformers_conf["model_class_name"])



    
def _load_model(path, model_class_name):
    """
    :param path: The path to a serialized PyTorch model.
    :param model_class_name: transformers model class name
    """
    import torch, transformers
    from transformers import AutoTokenizer


    tokenizer = None

    model_path = os.path.join(path, _MODEL_DIR_SUBPATH)
    tokenizer_path = os.path.join(path, _TOKENIZER_DIR_SUBPATH)

    try:
     model_class = getattr(transformers, model_class_name)
    except ImportError as exc:
        raise MlflowException(
            message=(
                "Failed to import the transformers model class for that model"
                " model `{model_class_name}` is not supported".format(
                    model_class_name=model_class_name
                )
            ),
            error_code=RESOURCE_DOES_NOT_EXIST,
        ) from exc

    model = model_class.from_pretrained(model_path)
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer
    return model










def log_tokenizer(tokenizer, artifact_path):
    """
    Log a tokenizer

    
    :param tokenizer: tokenizer to be saved.
    :param artifact_path: Run-relative artifact path.

    """
    import torch, transformers
    try:
        getattr(transformers, tokenizer.__class__.__name__)
    except ImportError as exc:
        raise MlflowException(
            message=(
                "Failed to import the transformers tokenizer class for that tokenizer"
                " tokenizer `{tok_class_name}` is not supported".format(
                    tok_class_name=tokenizer.__class__.__name__
                )
            ),
            error_code=RESOURCE_DOES_NOT_EXIST,
        ) from exc
    
    with TempDir() as tmp:
        local_path = tmp.path()
        save_tokenizer(tokenizer=tokenizer, path=local_path)
        mlflow.log_artifacts(local_path, artifact_path)


def save_tokenizer(tokenizer, path):
    """
    Save a tokenizer to a path on the local file system

    :param tokenizer: state_dict to be saved.
    :param path: Local path where the state_dict is to be saved.
    """
    import torch, transformers

    # The object type check here aims to prevent a scenario where a user accidentally passees
    # a model instead of a state_dict and `torch.save` (which accepts both model and state_dict)
    # successfully completes, leaving the user unaware of the mistake.
    try:
        getattr(transformers, tokenizer.__class__.__name__)
    except ImportError as exc:
        raise MlflowException(
            message=(
                "Failed to import the transformers tokenizer class for that tokenizer"
                " tokenizer `{tok_class_name}` is not supported".format(
                    tok_class_name=tokenizer.__class__.__name__
                )
            ),
            error_code=RESOURCE_DOES_NOT_EXIST,
        ) from exc
    
    os.makedirs(path, exist_ok=True)
    tokenizer.save_pretrained(path)


def load_tokenizer(tokenizer_uri):
    """
    Load a tokenizer from a local file or a run.

    :param stokenizer_uri: The location, in URI format, of the state_dict, for example:

                    - ``/Users/me/path/to/local/tokenizer``
                    - ``relative/path/to/local/tokenizer``
                    - ``s3://my_bucket/path/to/tokenizer``
                    - ``runs:/<mlflow_run_id>/run-relative/path/to/tokenizer``

                    For more information about supported URI schemes, see
                    `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                    artifact-locations>`_.
    :return: A tokenizer

    """
    import torch, transformers
    from transformers import AutoTokenizer

    tokenizer_path = _download_artifact_from_uri(artifact_uri=tokenizer_uri)

    return AutoTokenizer.from_pretrained(tokenizer_path)
