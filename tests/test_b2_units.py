import brian2 as b2
import numpy as np
import pytest

from neuroseq.b2_units import display_units
from neuroseq.b2_units import quantity_to_value_unit_str
from neuroseq.b2_units import str_to_quantity
from neuroseq.b2_units import value_unit_str_to_quantity


@pytest.mark.parametrize(
    ["value", "units_str", "quantity"],
    [
        (1.0, "mV", b2.mV),
        (1, "mV", b2.mV),
        (np.array([1, 2]), "mV", np.array([1, 2]) * b2.mV),
        (2, "pA", 2 * b2.pA),
        (3, "V", 3 * b2.volt),
        (2, "S", 2 * b2.siemens),
        (1, "A", 1 * b2.amp),
        (2, "s", 2 * b2.second),
        (1.0, "V/ms", b2.volt / b2.ms),
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


@pytest.mark.parametrize(
    "quantity",
    [
        3 * b2.amp,
        0.0002 * b2.ms,
        5e-12 * b2.ms,
        -2e-3 * b2.ms,
        -3 * b2.ms,
        1 / 10 * b2.ms,
        1.5 * 1e-5 * b2.ms,
        7 * b2.us,
        21.3 * b2.mS,
    ],
)
def test_quantity_parser_round_trip_quantities(quantity: b2.Quantity) -> None:
    string = str(quantity.in_best_unit(python_code=True))
    parsed_quantity = str_to_quantity(string)
    assert isinstance(parsed_quantity, b2.Quantity)
    assert parsed_quantity.has_same_dimensions(quantity)
    assert np.allclose(np.asarray(parsed_quantity), np.asarray(quantity))


def test_quantity_parser() -> None:
    # plain numeric literals come back as numbers, not strings
    assert str_to_quantity("3") == 3
    assert isinstance(str_to_quantity("3"), int)

    assert str_to_quantity("3 * ms") == 3 * b2.ms

    assert str_to_quantity("+3 * ms") == 3 * b2.ms

    assert str_to_quantity("-3 * ms") == -3 * b2.ms

    assert str_to_quantity("3e-1 * ms") == 0.3 * b2.ms

    assert str_to_quantity("3. * ms") == 3 * b2.ms

    assert str_to_quantity("3.  * ms  ") == 3 * b2.ms

    assert str_to_quantity("1/(10) * ms") == 1 / 10 * b2.ms

    assert str_to_quantity("35") == 35
    assert isinstance(str_to_quantity("35"), int)

    # a float literal round-trips as a float
    assert str_to_quantity("0.02") == 0.02
    assert isinstance(str_to_quantity("0.02"), float)

    # booleans stay strings (dropdown values like skip_if_invalid)
    assert str_to_quantity("True") == "True"
    assert str_to_quantity("False") == "False"

    assert str_to_quantity("x*y") == "x*y"

    assert str_to_quantity("1/(10) * Mx") == "1/(10) * Mx"

    assert str_to_quantity("euler") == "euler"

    assert str_to_quantity("exponential_euler") == "exponential_euler"

    assert str_to_quantity("exact") == "exact"

    assert str_to_quantity("rk2") == "rk2"

    assert str_to_quantity("rk4") == "rk4"

    assert str_to_quantity("heun") == "heun"

    assert str_to_quantity("milstein") == "milstein"

    assert str_to_quantity("2e10 * ms") == 2e10 * b2.ms

    assert str_to_quantity("2e-5 * ms") == 2e-5 * b2.ms
