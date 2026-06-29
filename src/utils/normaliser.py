import math


def normalise_score(value, p10, p90):
    """
    Min-max normalise using p10 as floor and p90 as ceiling.
    Clamp output to [0.0, 1.0].
    """
    if p90 <= p10:
        return 0.5
    clamped = max(p10, min(p90, value))
    return (clamped - p10) / (p90 - p10)


def experience_band_score(years: float, optimal: float = 5.1,
                          halfwidth: float = 4.0) -> float:
    """
    Asymmetric Gaussian — penalises under-experience harder than over.
    Under optimal: halfwidth = 2.5 (steeper decay)
    Over optimal:  halfwidth = 5.0 (gentler decay)
    """
    if years < optimal:
        return math.exp(-0.5 * ((years - optimal) / 2.5) ** 2)
    else:
        return math.exp(-0.5 * ((years - optimal) / 5.0) ** 2)
