from mlcf.utils import ListEnum
from enum import unique
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# MLCF modules
from mlcf.datatools.utils import build_forecast_ts_training_dataset
from mlcf.datatools.wtseries import WTSeries
from mlcf.envtools.hometools import MlcfHome


@unique
class Partition(ListEnum):
    TRAIN: str = "train"
    VALIDATION: str = "validation"
    TEST: str = "test"


@unique
class Field(ListEnum):
    INPUT: str = "input"
    TARGET: str = "target"


TRAIN: str = Partition.TRAIN.value
VALIDATION: str = Partition.VALIDATION.value
TEST: str = Partition.TEST.value
INPUT: str = Field.INPUT.value
TARGET: str = Field.TARGET.value


def get_enum_value(enum, ListEnum) -> str:
    return enum.value if isinstance(enum, ListEnum) else enum


class WTSTColumnIndexException(Exception):
    pass


class WTSTFeaturesException(Exception):
    pass


class WTSTraining(object):
    def __init__(
        self,
        input_width: int,
        target_width: int = 1,
        partition: Union[str, Partition] = Partition.TRAIN,
        features: List[str] = [],
        index_column: str = None,
        project: MlcfHome = None,
        *args,
        **kwargs,
    ):
        """WTSTraining allow to handle time series data in a machine learning training.
        The component of the WTSTraining is the WTSeries which is a list of window
        extract from window sliding of a time series data.

        Args:
            input_width (int): The number of available time / the input width for a ml model
            target_width (int, optional): the size of the target /
            the size of the output for a ml model. Defaults to 1.
            index_column (str, optional): the name of the column we want to index the data. In
            general it's "Date". Defaults to None.
        """
        self._init(features, partition, index_column)

        self.input_width: int = input_width
        self.target_width: int = target_width

        self.project: Optional[MlcfHome] = project

        self.train_data: Dict = {
            INPUT: WTSeries(self.input_width),
            TARGET: WTSeries(self.target_width),
        }

        self.val_data: Dict = {
            INPUT: WTSeries(self.input_width),
            TARGET: WTSeries(self.target_width),
        }

        self.test_data: Dict = {
            INPUT: WTSeries(self.input_width),
            TARGET: WTSeries(self.target_width),
        }

        self.ts_data: Dict = {
            TRAIN: self.train_data,
            VALIDATION: self.val_data,
            TEST: self.test_data,
        }

    def _init(self, features, partition, index_column):
        self.features_has_been_set = False
        self.index_column_has_been_set = False
        self.index_column: str = ""
        self.features: List[str] = []
        self.part_str: str
        self.set_partition(partition)
        if len(features) != 0:
            self.set_features(features)
        if index_column is not None:
            self.set_index_column(index_column)

    def set_features(self, features: List[str]):
        self.features = list(features)
        self.features_has_been_set = True

    def set_index_column(self, index_column: str):
        self.index_column = index_column
        self.index_column_has_been_set = True

    def set_partition(self, partition: Union[str, Partition]):
        self.part_str = get_enum_value(partition, Partition)

    def add_time_serie(
        self,
        dataframe: pd.DataFrame,
        *args, **kwargs
    ):
        """extend the time series data by extracting the window data from a input dataframe

        Args:
            dataframe (pd.DataFrame): A input raw dataframe
            prop_tv (float, optional): The percentage of test and val part. Defaults to 0.2.
            prop_v (float, optional): The percentage of val part in
            the union of test and val part. Defaults to 0.3.
            n_interval (int, optional): A number of interval to divide the raw data
            before windowing. Allow to homogenized the ts data. Defaults to 1.
            offset (int, optional): the width time between input and the target. Defaults to 0.
            window_step (int, optional): the step of each window. Defaults to 1.
        """
        data = dataframe.copy()
        if self.index_column_has_been_set:
            data.set_index(self.index_column, inplace=True)
        if self.features_has_been_set:
            selected_data = data[self.features]
        else:
            selected_data = data
            self.set_features(data.columns)
        if self.project:
            self.project.log.debug(
                f"[WTST]- Data length {len(selected_data)} will be add to the WTST data."
            )
        training_dataset: Tuple = build_forecast_ts_training_dataset(
            selected_data,
            self.input_width,
            self.target_width,
            *args, **kwargs
        )

        self.add_wtseries(*training_dataset)

    def add_wtseries(
        self,
        input_tr_data,
        target_tr_data,
        input_val_data,
        target_val_data,
        input_te_data,
        target_te_date
    ):
        self.add_one_pair_wtsdata(
            input_ts_data=input_tr_data,
            target_ts_data=target_tr_data,
            partition=Partition.TRAIN
        )

        self.add_one_pair_wtsdata(
            input_ts_data=input_val_data,
            target_ts_data=target_val_data,
            partition=Partition.VALIDATION
        )

        self.add_one_pair_wtsdata(
            input_ts_data=input_te_data,
            target_ts_data=target_te_date,
            partition=Partition.TEST
        )
        if self.project:
            self.project.log.debug(f"[WTST]- Add WTSeries data: {self}")

    def get(
        self,
        part: Union[str, Partition],
        field: Union[str, Field]
    ) -> WTSeries:
        part_str = get_enum_value(part, Partition)
        field_str = get_enum_value(field, Field)
        return self.ts_data[part_str][field_str]

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        inputs, targets = self()
        return inputs[idx], targets[idx]

    def __call__(
        self,
        part: Union[Partition, str] = None
    ) -> Tuple[WTSeries, WTSeries]:
        """return the time series data (a dict format) if None arguments has been filled.
        If part is filled, return the partition (train, validation, or test) (with a dict format).
        If field is filled, return the field (input or target) window data

        Args:
            part (Partition, optional): The partition ("train", "validation" or "test") we want to
            return.
            Defaults to None.
            field (Field, optional): The field ("input", or "target") we want to return.
            Defaults to None.

        Raises:
            Exception: You should fill part if field is filled

        Returns:
            Union[Dict, Dict[WTSeries], WTSeries]:
            A dict of Dict of window data (all the time series data),
            or a dict of window data (a part 'train', 'validation' or 'test'),
            or a window data (a field 'input', 'target')
        """
        if part is not None:
            part_str = get_enum_value(part, Partition)
            return (
                self.get(part_str, Field.INPUT.value),
                self.get(part_str, Field.TARGET.value)
            )
        return (
            self.get(self.part_str, Field.INPUT.value),
            self.get(self.part_str, Field.TARGET.value)
        )

    def __len__(self) -> int:
        return self.len()

    def len(self, part: Union[str, Partition] = None) -> int:
        inputs, _ = self(part)
        return len(inputs)

    def width(self) -> Tuple[int, int]:
        """return the width of windows data given 'input' or 'target'

        Args:
            field (Field, optional): 'input' or 'target'. Defaults to None.

        Raises:
            ValueError: Only 'input' or 'target' are allowed

        Returns:
            Union[int, Tuple[int, int]]: (input or target width)
            or respectively both if field is None
        """
        return self.input_width, self.target_width

    def ndim(self) -> int:
        if self.features_has_been_set:
            return len(self.features)
        return 0

    def copy(
        self,
        filter: Optional[List[Union[bool, str]]] = None
    ):
        wtstraining_copy = WTSTraining(
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
        wtstraining_copy.train_data = {
            INPUT: self.train_data[INPUT].copy(filter),
            TARGET: self.train_data[TARGET].copy(filter),
        }
        wtstraining_copy.val_data = {
            INPUT: self.val_data[INPUT].copy(filter),
            TARGET: self.val_data[TARGET].copy(filter),
        }
        wtstraining_copy.test_data = {
            INPUT: self.test_data[INPUT].copy(filter),
            TARGET: self.test_data[TARGET].copy(filter),
        }
        wtstraining_copy.ts_data = {
            TRAIN: wtstraining_copy.train_data,
            VALIDATION: wtstraining_copy.val_data,
            TEST: wtstraining_copy.test_data,
        }
        return wtstraining_copy

    def __str__(self) -> str:
        return (
            f"Input size: {self.input_width}, Target size: {self.target_width}, "
            + f"Index name: '{self.index_column if self.index_column_has_been_set else 'X'}' "
            + f"Data lengths: Train: {self.len(Partition.TRAIN)}, "
            + f"Validation: {self.len(Partition.VALIDATION)}, "
            + f"Test: {self.len(Partition.TEST)}"
        )

    def check_wtsdata(self, wtsdata: WTSeries):
        data = wtsdata.copy()

        if self.index_column_has_been_set and data[0].index.name != self.index_column:
            raise WTSTColumnIndexException(
                f"The WTSeries index : {data[0].index.name} is not equal " +
                f"to the WTSTraining index : {self.index_column}")

        if self.features_has_been_set and list(data[0].columns) != self.features:
            raise WTSTFeaturesException(
                f"The WTSeries features : {list(data[0].columns)} are " +
                f"not equals to the WTSTraining features : {self.features}")
        return True

    def add_one_pair_wtsdata(
        self,
        input_ts_data: WTSeries,
        target_ts_data: WTSeries,
        partition: Partition,
    ):
        assert self.check_wtsdata(input_ts_data)
        assert self.check_wtsdata(target_ts_data)
        self._add_wts_data(input_ts_data, target_ts_data, partition)

    def _add_wts_data(
        self,
        input_ts_data: WTSeries,
        target_ts_data: WTSeries,
        partition: Partition,
    ):
        """_add_ts_data add a Input ts data and a target ts data to the train, val or test part.
        In function of the {partition} parameter which is the name of the part
        (train, validation or test)

        Args:
            input_ts_data (WTSeries): A window data refferring to the input data
            target_ts_data (WTSeries): A window data refferring to the target data
            partition (Partition): the name of the part: 'train', 'validation' or 'test'
            do_shuffle (bool, optional): perform a shuffle if True. Defaults to False.
        """
        inputs, targets = self(partition)
        inputs.add_window_data(input_ts_data, ignore_data_empty=True)
        targets.add_window_data(target_ts_data, ignore_data_empty=True)
