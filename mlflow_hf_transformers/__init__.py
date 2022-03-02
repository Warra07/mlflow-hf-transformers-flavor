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
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
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
                # We include CloudPickle in the default environment because
                # it's required by the default pickle module used by `save_model()`
                # and `log_model()`: `mlflow.pytorch.pickle_module`.
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
def get_default_conda_env(style):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """

    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["altair==4.1.0", f"mlflow.styles.{style}"],
        additional_conda_channels=None,
    )


def log_model(
    model,
    artifact_path,
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
    Log a PyTorch model as an MLflow artifact for the current run.

        .. warning::

            Log the model with a signature to avoid inference errors.
            If the model is logged without a signature, the MLflow Model Server relies on the
            default inferred data type from NumPy. However, PyTorch often expects different
            defaults, particularly when parsing floats. You must include the signature to ensure
            that the model is logged with the correct data type so that the MLflow model server
            can correctly provide valid input.

    :param pytorch_model: PyTorch model to be saved. Can be either an eager model (subclass of
                          ``torch.nn.Module``) or scripted model prepared via ``torch.jit.script``
                          or ``torch.jit.trace``.

                          The model accept a single ``torch.FloatTensor`` as
                          input and produce a single output tensor.

                          If saving an eager model, any code dependencies of the
                          model's class, including the class definition itself, should be
                          included in one of the following locations:

                          - The package(s) listed in the model's Conda environment, specified
                            by the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_paths`` parameter.

    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param pickle_module: The module that PyTorch should use to serialize ("pickle") the specified
                          ``pytorch_model``. This is passed as the ``pickle_module`` parameter
                          to ``torch.save()``. By default, this module is also used to
                          deserialize ("unpickle") the PyTorch model at load time.
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
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

        .. code-block:: python

            requirements_file = "s3://my-bucket/path/to/my_file"

        In this case, the ``"my_file"`` requirements file is downloaded from S3. If ``None``,
        no requirements file is added to the model.

    :param extra_files: A list containing the paths to corresponding extra files. Remote URIs
                      are resolved to absolute filesystem paths.
                      For example, consider the following ``extra_files`` list -

                      extra_files = ["s3://my-bucket/path/to/my_file1",
                                    "s3://my-bucket/path/to/my_file2"]

                      In this case, the ``"my_file1 & my_file2"`` extra file is downloaded from S3.

                      If ``None``, no extra files are added to the model.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to ``torch.save`` method.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import numpy as np
        import torch
        import mlflow.pytorch

        class LinearNNModel(torch.nn.Module):
            def __init__(self):
                super(LinearNNModel, self).__init__()
                self.linear = torch.nn.Linear(1, 1)  # One in and one out

            def forward(self, x):
                y_pred = self.linear(x)
                return y_pred

        def gen_data():
            # Example linear model modified to use y = 2x
            # from https://github.com/hunkim/PyTorchZeroToAll
            # X training data, y labels
            X = torch.arange(1.0, 25.0).view(-1, 1)
            y = torch.from_numpy(np.array([x * 2 for x in X])).view(-1, 1)
            return X, y

        # Define model, loss, and optimizer
        model = LinearNNModel()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # Training loop
        epochs = 250
        X, y = gen_data()
        for epoch in range(epochs):
            # Forward pass: Compute predicted y by passing X to the model
            y_pred = model(X)

            # Compute the loss
            loss = criterion(y_pred, y)

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

            # convert to scripted model and log the model
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

        # Fetch the logged model artifacts
        print("run_id: {}".format(run.info.run_id))
        for artifact_path in ["model/data", "scripted_model/data"]:
            artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id,
                        artifact_path)]
            print("artifacts: {}".format(artifacts))

    .. code-block:: text
        :caption: Output

        run_id: 1a1ec9e413ce48e9abf9aec20efd6f71
        artifacts: ['model/data/model.pth',
                    'model/data/pickle_module_info.txt']
        artifacts: ['scripted_model/data/model.pth',
                    'scripted_model/data/pickle_module_info.txt']

    .. figure:: ../_static/images/pytorch_logged_models.png

        PyTorch logged models
    """
    import transformers
    
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_hf_transformers,
        model=model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        requirements_file=requirements_file,
        extra_files=extra_files,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
    )



def save_model(
    model,
    path,
    conda_env=None,
    mlflow_model=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    requirements_file=None,
    extra_files=None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """Save a visual model to a local file or a run.


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

    if requirements_file:

        warnings.warn(
            "`requirements_file` has been deprecated. Please use `pip_requirements` instead.",
            FutureWarning,
            stacklevel=2,
        )

        if not isinstance(requirements_file, str):
            raise TypeError("Path to requirements file should be a string")

        with TempDir() as tmp_requirements_dir:
            _download_artifact_from_uri(
                artifact_uri=requirements_file, output_path=tmp_requirements_dir.path()
            )
            rel_path = os.path.basename(requirements_file)
            torchserve_artifacts_config[_REQUIREMENTS_FILE_KEY] = {"path": rel_path}
            shutil.move(tmp_requirements_dir.path(rel_path), path)

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


    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                model_data_path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    if not requirements_file:
        # Save `requirements.txt`
        write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))




def load_model(model_uri, dst_path=None, **kwargs):
    """
    Load a PyTorch model from a local file or a run.

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

    :param kwargs: kwargs to pass to ``torch.load`` method.
    :return: A PyTorch model.

    .. code-block:: python
        :caption: Example

        import torch
        import mlflow.pytorch

        # Class defined here
        class LinearNNModel(torch.nn.Module):
            ...

        # Initialize our model, criterion and optimizer
        ...

        # Training loop
        ...

        # Log the model
        with mlflow.start_run() as run:
            mlflow.pytorch.log_model(model, "model")

        # Inference after loading the logged model
        model_uri = "runs:/{}/model".format(run.info.run_id)
        loaded_model = mlflow.pytorch.load_model(model_uri)
        for x in [4.0, 6.0, 30.0]:
            X = torch.Tensor([[x]])
            y_pred = loaded_model(X)
            print("predict X: {}, y_pred: {:.2f}".format(x, y_pred.data.item()))

    .. code-block:: text
        :caption: Output

        predict X: 4.0, y_pred: 7.57
        predict X: 6.0, y_pred: 11.64
        predict X: 30.0, y_pred: 60.48
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
    :param kwargs: Additional kwargs to pass to the PyTorch ``torch.load`` function.
    """
    import torch, transformers


    model_path = os.path.join(path, _MODEL_DIR_SUBPATH)

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
    
    return model_class.from_pretrained(model_path)








def log_tokenizer(tokenizer, artifact_path):
    """
    Log a state_dict as an MLflow artifact for the current run.

    .. warning::
        This function just logs a state_dict as an artifact and doesn't generate
        an :ref:`MLflow Model <models>`.

    :param state_dict: state_dict to be saved.
    :param artifact_path: Run-relative artifact path.
    :param kwargs: kwargs to pass to ``torch.save``.

    .. code-block:: python
        :caption: Example

        # Log a model as a state_dict
        with mlflow.start_run():
            state_dict = model.state_dict()
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="model")

        # Log a checkpoint as a state_dict
        with mlflow.start_run():
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
            }
            mlflow.pytorch.log_state_dict(state_dict, artifact_path="checkpoint")
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
        save_state_dict(tokenizer=tokenizer, path=local_path)
        mlflow.log_artifacts(local_path, artifact_path)


def save_tokenizer(tokenizer, path):
    """
    Save a state_dict to a path on the local file system

    :param state_dict: state_dict to be saved.
    :param path: Local path where the state_dict is to be saved.
    :param kwargs: kwargs to pass to ``torch.save``.
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
    tokenizer_path = os.path.join(path, _TOKENIZER_DIR_SUBPATH)
    tokenizer.save_prentrained(tokenizer_path)


def load_tokenizer(tokenizer_uri):
    """
    Load a state_dict from a local file or a run.

    :param state_dict_uri: The location, in URI format, of the state_dict, for example:

                    - ``/Users/me/path/to/local/state_dict``
                    - ``relative/path/to/local/state_dict``
                    - ``s3://my_bucket/path/to/state_dict``
                    - ``runs:/<mlflow_run_id>/run-relative/path/to/state_dict``

                    For more information about supported URI schemes, see
                    `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                    artifact-locations>`_.

    :param kwargs: kwargs to pass to ``torch.load``.
    :return: A state_dict

    .. code-block:: python
        :caption: Example

        with mlflow.start_run():
            artifact_path = "model"
            mlflow.pytorch.log_state_dict(model.state_dict(), artifact_path)
            state_dict_uri = mlflow.get_artifact_uri(artifact_path)

        state_dict = mlflow.pytorch.load_state_dict(state_dict_uri)
    """
    import torch, transformers
    try:
        tokenizer = getattr(transformers, tokenizer.__class__.__name__)
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
    
    tokenizer_path = _download_artifact_from_uri(artifact_uri=tokenizer_uri)

    return tokenizer.load(tokenizer_path)
