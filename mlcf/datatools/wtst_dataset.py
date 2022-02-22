
from pathlib import Path
from typing import List, Optional, Union

import zipfile as z
import os
import pandas as pd

from mlcf.datatools.wtst import Partition, Field, WTSTraining, get_partition_value
from mlcf.datatools.wtseries import WTSeries

EXTENSION_FILE = ".wtst"
NUMBER_OF_WINDOWS = 10000  # to pack in one folder when we write
TS_DATA_ARCHDIR: Path = "TS_DATA"


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


def add_wtseries_in_dataset(
    inputs: WTSeries,
    targets: WTSeries,
    dataset_path: Path,
    partition: Union[str, Partition]
):
    current_idx: int = 0
    part_str: str = get_partition_value(partition)
    dataset_path = dataset_path.with_suffix(EXTENSION_FILE)
    if dataset_path.is_file():
        zip_path: z.Path = z.Path(dataset_path)
        part_dir: z.Path = zip_path.joinpath(TS_DATA_ARCHDIR, part_str)
        max_packet = max([int(packet) for packet in part_dir.iterdir()])
        data_dir: z.Path = part_dir.joinpath(max_packet, "inputs")
        current_idx = max([int(str(file.stem).split("_")[-2]) for file in data_dir.iterdir()])

    with z.ZipFile(dataset_path, "w") as zipf:
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
                curr_dir=dataset_path.parent,
                arch_path=arch_x_path,
            )
            add_dataframe_to_zipfile(
                dataframe=y_window,
                zipf=zipf,
                curr_dir=dataset_path.parent,
                arch_path=arch_y_path
            )


class WTSTrainingDataset(WTSTraining):
    def __init__(
        self,
        dataset_path: Path,
        input_width: int = None,
        target_width: int = None,
        features: List[str] = [],
        index_column: str = None,
        partition: Optional[Union[str, Partition]] = "train"
    ):
        self.dataset_path: Path = dataset_path.with_suffix(EXTENSION_FILE)
        self.indexes = {
            Partition.TRAIN: 0,
            Partition.VALIDATION: 0,
            Partition.TEST: 0
        }
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
            partition=partition
        )

    def __getitem__(self, idx):
        window_inputs: pd.DataFrame
        window_targets: pd.DataFrame
        packet: int = (idx // NUMBER_OF_WINDOWS) * NUMBER_OF_WINDOWS
        with z.ZipFile(self.dataset_path, "r") as zipf:
            # traiter les données (date en index par ex)
            window_inputs = pd.read_csv(zipf.open(get_arch_path(
                packet=packet,
                idx=idx,
                part_str=self.part_str,
                field=Field.INPUT
            )))
            window_targets = pd.read_csv(zipf.open(get_arch_path(
                packet=packet,
                idx=idx,
                part_str=self.part_str,
                field=Field.TARGET
            )))
        return window_inputs, window_targets
