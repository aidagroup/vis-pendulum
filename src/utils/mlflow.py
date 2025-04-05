import mlflow
import numpy as np
import os
import sys

from typing import Dict, Any, Tuple, Union, Optional, List
from stable_baselines3.common.logger import (
    HumanOutputFormat,
    KVWriter,
    Logger,
    INFO,
)
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from src import run_path


@dataclass
class MlflowConfig:
    experiment_name: str
    tracking_uri: str = "file://" + os.path.join(str(run_path), "mlruns")
    run_name: Optional[str] = None


class MLflowOutputFormat(KVWriter):
    """Dumps key/value pairs into MLflow's numeric format."""

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class SilentLogger(Logger):
    def __init__(
        self,
        folder: Optional[str] = None,
        output_formats: Optional[List[KVWriter]] = None,
    ):
        self.name_to_value = defaultdict(
            float
        )  # Preserve the original Logger attributes
        self.name_to_count = defaultdict(int)
        self.name_to_excluded = defaultdict(
            lambda: None
        )  # Use a dictionary for exclusions
        self.level = INFO
        self.folder = folder
        self.output_formats = output_formats or []

        if folder is not None:
            os.makedirs(folder, exist_ok=True)


def create_mlflow_logger():
    logger = SilentLogger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
    )
    return logger


def mlflow_monitoring():
    def inner1(func):
        def inner2(*args, **kwargs):
            mlflow_config: MlflowConfig = args[0].mlflow
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            mlflow.set_experiment(mlflow_config.experiment_name)

            # print("run_name:", run_name)
            with mlflow.start_run(run_name=mlflow_config.run_name):
                # log param
                if len(args):
                    args_dict = vars(args[0])
                    [
                        mlflow.log_param(k, args_dict[k])
                        for k in args_dict
                        if k != "mlflow"
                    ]

                return func(*args, **kwargs)

        return inner2

    return inner1
