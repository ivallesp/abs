from unittest import TestCase
from src.models import get_model
from functools import partial


class TestGetModelFunction(TestCase):
    def test_model_not_found_exception(self):
        f = partial(
            get_model,
            name="notexistingname",
            params={"activation_name": "relu", "n_outputs": 10},
        )
        self.assertRaises(ModuleNotFoundError, f)

    def test_getting_test_model(self):
        model, loss, optimizer = get_model(
            name="TestNet", params={"activation_name": "relu", "n_outputs": 10}
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(loss)
        self.assertIsNotNone(optimizer)
