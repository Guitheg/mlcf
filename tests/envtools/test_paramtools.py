from mlcf.envtools.paramtools import get_param_dict_from_str


def test_get_param_dict_from_str():
    args = ["a=1", "b=2,3", "c=4"]
    dict = get_param_dict_from_str(args)
    assert dict["a"] == "1"
    assert dict["b"] == ["2", "3"]
