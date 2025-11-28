import brian2 as b2
import numpy as np
import pytest

from neuroseq.b2_units import display_units
from neuroseq.b2_units import quantity_to_value_unit_str
from neuroseq.b2_units import value_unit_str_to_quantity


@pytest.mark.parametrize(
    ["value", "units_str", "quantity"],
    [
        (1.0, "mV", b2.mV),
        (1, "mV", b2.mV),
        (np.array([1, 2]), "mV", np.array([1, 2]) * b2.mV),
        (2, "pA", 2 * b2.pA),
        (2, "metre**2", 2 * b2.metre * b2.metre),
        (2, "metre*metre", 2 * b2.metre**2),
        (2, "metre**3", 2 * b2.metre**3),
        (1.0, "volt/ms", b2.volt / b2.ms),
        (0.01, "volt/ms", 10 * b2.volt / b2.second),
        (3, "volt", 3 * b2.volt),
        (1, "0.5*amp", 0.5 * b2.amp),
        (5, "2*ms", 10 * b2.ms),
        (2, "second", 2 * b2.second),
        (1.5, "Hz", 1.5 * b2.Hz),
        (1, "1.5/second", 1.5 * b2.Hz),
        (1000, "Hz", 1000 * b2.Hz),
        (1000, "Hz", b2.kHz),
        (1, "siemens", b2.siemens),
        (1e-9, "siemens", 1e-9 * b2.siemens),
        (1e-9, "siemens", b2.nS),
        (1, "nS", 1e-9 * b2.siemens),
        (1.00, "nS", b2.nS),
    ],
)
def test_b2_str_to_quantity(
    value: int | float | np.ndarray, units_str: str, quantity: b2.Quantity
):
    # check that the string is correctly converted to the expected quantity
    new_quantity = value_unit_str_to_quantity(value, units_str)
    if isinstance(value, np.ndarray):
        np.testing.assert_allclose(new_quantity, quantity)
    else:
        assert new_quantity == quantity
    # check that converting this quantity back to string and then back to quantity gives the same quantity
    new_value, new_units_str = quantity_to_value_unit_str(new_quantity)
    round_trip_quantity = value_unit_str_to_quantity(new_value, new_units_str)
    if isinstance(value, np.ndarray):
        np.testing.assert_allclose(round_trip_quantity, quantity)
    else:
        assert round_trip_quantity == quantity


@pytest.mark.parametrize(
    ["quantity", "units"],
    [
        (15 * b2.mV, "V"),
        (1e6 * b2.mV, "V"),
        (2 * b2.nS, "S"),
        (5000 * b2.Hz, "Hz"),
        (0.5 * b2.kHz, "Hz"),
        (2000 * b2.ms, "s"),
    ],
)
def test_display_units(quantity: b2.Quantity, units: str):
    assert display_units(quantity) == units
