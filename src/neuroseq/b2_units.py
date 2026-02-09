import logging

import brian2 as b2
import brian2.numpy_ as np

# these import * statements are needed so that we can do eval("<brian2 Quantity expression as python code>"):
from brian2 import *  # noqa: F403
from brian2.units.allunits import *  # noqa: F403


def quantity_to_value_unit_str(
    quantity: b2.Quantity,
) -> tuple[int | float | np.ndarray, str]:
    # note: there is no way to get the exact original units back from a Quantity, since
    # e.g., 1000*Hz and 1*kHz are equivalent and both will be converted to 1 kilohertz internally,
    # but we can guarantee that the returned string can be eval'ed back to the same Quantity

    # get values without units
    value = b2.asarray(quantity)
    if value.shape == ():
        value = value.item()

    # get units as a string that can be eval'ed back to a Quantity
    units = b2.Quantity(1, dim=quantity.dim).in_best_unit(python_code=True)

    return value, units


def value_unit_str_to_quantity(
    value: int | float | np.ndarray, unit_str: str
) -> b2.Quantity:
    return eval(unit_str) * value


def str_to_quantity(string: str) -> b2.Quantity | str:
    try:
        quantity = eval(string)
        if isinstance(quantity, b2.Quantity):
            return quantity
        else:
            logging.warning(
                f"'{string}' parsed to type {type(quantity)}, not b2.Quantity, returning original string"
            )
            return string
    except Exception:
        logging.warning(
            f"Could not parse '{string}' as b2.Quantity, returning original string"
        )
        return string


def display_units(quantity: b2.Quantity) -> str:
    # Handle the case where quantity might not be a proper Brian2 Quantity
    if not isinstance(quantity, b2.Quantity):
        # If it's a plain number or array (dimensionless), return empty string
        return ""

    try:
        unit_str = b2.Quantity(1, dim=quantity.dim).in_best_unit(python_code=False)
        if unit_str[0:3] == "1. ":
            unit_str = unit_str[3:]  # remove leading "1. " for dimensionless units
        return unit_str
    except (AttributeError, TypeError):
        # If we can't get dimensions, assume dimensionless
        return ""
