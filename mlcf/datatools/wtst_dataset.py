
from pathlib import Path
from typing import List, Optional, Union, Tuple

import zipfile as z
import os
import pandas as pd
from os.path import relpath

from mlcf.datatools.wtst import Partition, Field, WTSTraining, get_enum_value
from mlcf.datatools.wtseries import WTSeries
from mlcf.envtools.hometools import MlcfHome

EXTENSION_FILE = ".wtst"
NUMBER_OF_WINDOWS = 10000  # to pack in one folder when we write
TS_DATA_ARCHDIR = "WTSTrainingDataset"


def is_dir_in_zipfile(zipf: z.ZipFile, path: Path):
    for info in zipf.infolist():
        if info.filename.startswith(str(path)):
            return True
    return False


def iterdir_in_zipfile(zipf: z.ZipFile, path: Path):
    return {
        relpath(
            info.filename, str(path)).split(os.sep)[0]
        for info in zipf.infolist()
        if info.filename.startswith(str(str(path)))
    }


def get_arch_path(packet: str, idx: int, part_str: str, field_str: str) -> Path:
    return Path(TS_DATA_ARCHDIR).joinpath(
        part_str.swapcase(),
        str(packet),
        ('inputs' if field_str == 'input' else 'targets'),
        f"{part_str[:2]}_window_{idx}_{('X' if field_str == 'input' else 'y')}")


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
) -> pd.DataFrame:
    with zipf.open(str(get_arch_path(
        packet=str((idx // NUMBER_OF_WINDOWS)*NUMBER_OF_WINDOWS),
        idx=idx,
        part_str=part_str,
        field_str=field_str))
    ) as dataframe_file:
        window = pd.read_csv(dataframe_file)
    return window


class WTSTrainingDataset(WTSTraining):
    def __init__(
        self,
        dataset_path: Path,
        input_width: int = None,
        target_width: int = None,
        features: List[str] = [],
        index_column: str = None,
        partition: Union[str, Partition] = Partition.TRAIN,
        project: MlcfHome = None
    ):
        self.dataset_path: Path = dataset_path.with_suffix(EXTENSION_FILE)
        self.indexes = {
            Partition.TRAIN.value: 0,
            Partition.VALIDATION.value: 0,
            Partition.TEST.value: 0
        }
        self._init(features, partition, index_column)
        self.start_idx = 0
        if self.dataset_path.is_file():
            with z.ZipFile(self.dataset_path, "r") as zipf:
                for part in [
                    Partition.TRAIN.value.swapcase(),
                    Partition.VALIDATION.value.swapcase(),
                    Partition.TEST.value.swapcase()
                ]:
                    part_dir: Path = Path(TS_DATA_ARCHDIR).joinpath(part)

                    list_packet = [
                        int(packet) for packet
                        in iterdir_in_zipfile(zipf, part_dir)
                    ]
                    if len(list_packet) > 0:
                        max_packet = max(list_packet)
                        data_dir: Path = part_dir.joinpath(str(max_packet), "inputs")
                        self.indexes[part.lower()] = max([
                            int(str(file).split("_")[-2]) for file
                            in iterdir_in_zipfile(zipf, data_dir)
                        ]) + 1
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
        part: Union[str, Partition],
        field: Union[str, Field]
    ) -> WTSeries:
        part_str = get_enum_value(part, Partition)
        field_str = get_enum_value(field, Field)
        wtseries = WTSeries(
            window_width=self.input_width if field_str == "input" else self.target_width
        )
        with z.ZipFile(self.dataset_path, "r") as zipf:
            for idx in range(self.start_idx, self.start_idx+NUMBER_OF_WINDOWS):
                wtseries.add_one_window(
                    get_window_from_zipfile(
                        zipf=zipf,
                        idx=idx,
                        part_str=part_str,
                        field_str=field_str,
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
            window_inputs = get_window_from_zipfile(
                zipf=zipf,
                idx=idx,
                part_str=self.part_str,
                field_str=Field.INPUT.value,
                index_column=self.index_column,
                features=self.features
            )
            if self.index_column:
                window_inputs.set_index(self.index_column, inplace=True)
            window_targets = get_window_from_zipfile(
                zipf=zipf,
                idx=idx,
                part_str=self.part_str,
                field_str=Field.TARGET.value,
                index_column=self.index_column,
                features=self.features
            )
            if self.index_column:
                window_targets.set_index(self.index_column, inplace=True)
        return window_inputs, window_targets

    def len(self, part: Union[str, Partition] = None) -> int:
        if part:
            part_str = get_enum_value(part, Partition)
            return self.indexes[part_str]
        return self.indexes[self.part_str]

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
        input_ts_data: WTSeries,
        target_ts_data: WTSeries,
        partition: Union[str, Partition],
        *args, **kwargs
    ):
        part_str: str = get_enum_value(partition, Partition)
        current_idx: int = self.indexes[part_str]
        with z.ZipFile(self.dataset_path, "a") as zipf:
            for idx, (x_window, y_window) in enumerate(
                start=current_idx,
                iterable=zip(input_ts_data, target_ts_data)
            ):
                packet = str((idx // NUMBER_OF_WINDOWS)*NUMBER_OF_WINDOWS)
                arch_x_path = get_arch_path(packet, idx, part_str, Field.INPUT.value)
                arch_y_path = get_arch_path(packet, idx, part_str, Field.TARGET.value)
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
                self.indexes[part_str] += 1
        if self.project:
            self.project.log.info(
                f"Current length for {part_str} : {self.indexes[part_str]}"
            )
