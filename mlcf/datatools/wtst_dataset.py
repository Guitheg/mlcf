
from pathlib import Path
from typing import List, Optional, Union, Tuple

import zipfile as z
import os
import pandas as pd
from mlcf.datatools.preprocessing import Identity

from mlcf.datatools.wtst import Partition, Field, WTSTraining, get_enum_value
from mlcf.datatools.utils import build_forecast_ts_training_dataset
from mlcf.datatools.wtseries import WTSeries
from mlcf.envtools.hometools import MlcfHome

EXTENSION_FILE = ".wtst"
NUMBER_OF_WINDOWS = 10000  # to pack in one folder when we write
TS_DATA_ARCHDIR: Path = "WTSTrainingDataset"


def get_arch_path(packet: int, idx: int, part_str: str, field: Field) -> Path:
    return TS_DATA_ARCHDIR.joinpath(
        part_str.swapcase(),
        packet,
        ('inputs' if field.value == 'input' else 'targets'),
        f"{part_str[:2]}_window_{idx}_{('X' if field.value == 'input' else 'y')}")


def add_dataframe_to_zipfile(
    dataframe: pd.DataFrame,
    zipf: z.ZipFile,
    curr_dir: Path,
    arch_path: Path
):
    filepath = curr_dir.joinpath(arch_path.name)
    dataframe.to_csv(filepath)
    zipf.write(filepath, arch_path)
    os.remove(filepath)


def get_window_from_zipfile(
    zipf: z.ZipFile,
    idx: int,
    part_str: str,
    field_str: str,
    index_column: str,
    features: List[str]
):
    window = pd.read_csv(zipf.open(get_arch_path(
        packet=str((idx // NUMBER_OF_WINDOWS)*NUMBER_OF_WINDOWS),
        idx=idx,
        part_str=part_str,
        field=field_str
    )))
    return window


class WTSTrainingDataset(WTSTraining):
    def __init__(
        self,
        dataset_path: Path,
        input_width: int = None,
        target_width: int = None,
        features: List[str] = [],
        index_column: str = None,
        partition: Optional[Union[str, Partition]] = "train",
        project: MlcfHome = None
    ):
        self.dataset_path: Path = dataset_path.with_suffix(EXTENSION_FILE)
        self.indexes = {
            Partition.TRAIN: 0,
            Partition.VALIDATION: 0,
            Partition.TEST: 0
        }
        self.start_idx = 0
        if dataset_path.is_file():
            zip_path: z.Path = z.Path(dataset_path)
            for part in [Partition.TRAIN.value, Partition.VALIDATION.value, Partition.TEST.value]:
                part_dir: z.Path = zip_path.joinpath(TS_DATA_ARCHDIR, part)
                max_packet = max([int(packet) for packet in part_dir.iterdir()])
                data_dir: z.Path = part_dir.joinpath(max_packet, "inputs")
                self.indexes[part] = max(
                    [int(str(file.stem).split("_")[-2]) for file in data_dir.iterdir()]
                )
                inputs, targets = self[0]
                features = inputs.columns
                input_width = len(inputs)
                target_width = len(targets)

        if input_width is None or target_width is None:
            raise Exception(
                "input width or target width are None. Give an input" +
                "and a target width OR give a existing dataset file path.")
        super(WTSTrainingDataset, self).__init__(
            input_width=input_width,
            target_width=target_width,
            features=features,
            index_column=index_column,
            partition=partition,
            project=project
        )

    def set_start_idx(self, idx: int):
        self.start_idx = idx

    def get(
        self,
        part: Partition,
        field: Field
    ) -> WTSeries:
        part_str = get_enum_value(part)
        field_str = get_enum_value(field)
        wtseries = WTSeries(
            input_width=self.input_width if field_str == "input" else self.target_width
        )
        with z.ZipFile(self.dataset_path, "r") as zipf:
            for idx in range(self.start_idx, self.start_idx+NUMBER_OF_WINDOWS):
                wtseries.add_one_window(
                    get_window_from_zipfile(
                        zipf=zipf,
                        idx=idx,
                        part_str=part_str,
                        field=field_str,
                        index_column=self.index_column,
                        features=self.features
                    )
                )
        self.set_start_idx(self.start_idx+NUMBER_OF_WINDOWS)
        return wtseries

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        window_inputs: pd.DataFrame
        window_targets: pd.DataFrame
        with z.ZipFile(self.dataset_path, "r") as zipf:
            # traiter les données (date en index par ex)
            window_inputs = get_window_from_zipfile(
                zipf=zipf,
                idx=idx,
                part_str=self.part_str,
                field=Field.INPUT,
                index_column=self.index_column,
                features=self.features
            )
            window_targets = get_window_from_zipfile(
                zipf=zipf,
                idx=idx,
                part_str=self.part_str,
                field=Field.TARGET,
                index_column=self.index_column,
                features=self.features
            )
        return window_inputs, window_targets

    def __call__(
        self,
        part: Union[Partition, str] = None
    ) -> Tuple[WTSeries, WTSeries]:

        if part is not None:
            part_str = get_enum_value(part)
            return (
                self.ts_data[part_str][Field.INPUT.value],
                self.ts_data[part_str][Field.TARGET.value]
            )
        return (
            self.ts_data[self.part_str][Field.INPUT.value],
            self.ts_data[self.part_str][Field.TARGET.value]
        )

    def len(self, part: Union[str, Partition] = None) -> int:
        part_str = get_enum_value(part)
        return self.indexes[part_str]

    def copy(
        self,
        filter: Optional[List[Union[bool, str]]] = None
    ):
        wtst_dataset = WTSTrainingDataset(
            self.dataset_path,
            input_width=self.input_width,
            target_width=self.target_width,
            features=(
                pd.DataFrame(columns=self.features).loc[:, filter].columns
                if filter else self.features
            ),
            index_column=self.index_column,
            partition=self.part_str,
            project=self.project
        )
        return wtst_dataset

    def __str__(self) -> str:
        return f"Dataset path : {self.dataset_path}. " + super(WTSTrainingDataset, self).__str__()

    def _add_wts_data(
        self,
        inputs: WTSeries,
        targets: WTSeries,
        partition: Union[str, Partition],
        *args, **kwargs
    ):
        part_str: str = get_enum_value(partition)
        current_idx: int = self.indexes[part_str]
        with z.ZipFile(self.dataset_path, "w") as zipf:
            for idx, (x_window, y_window) in enumerate(
                start=current_idx,
                iterable=zip(
                    inputs,
                    targets)
            ):
                packet = str((idx // NUMBER_OF_WINDOWS)*NUMBER_OF_WINDOWS)
                arch_x_path = get_arch_path(packet, idx, part_str, Field.INPUT)
                arch_y_path = get_arch_path(packet, idx, part_str, Field.TARGET)
                add_dataframe_to_zipfile(
                    dataframe=x_window,
                    zipf=zipf,
                    curr_dir=self.dataset_path.parent,
                    arch_path=arch_x_path,
                )
                add_dataframe_to_zipfile(
                    dataframe=y_window,
                    zipf=zipf,
                    curr_dir=self.dataset_path.parent,
                    arch_path=arch_y_path
                )
