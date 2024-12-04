# © Copyright 2021, PRISMA’s Authors

"""Function generators according to a linehsape.
Each function generator constructs lambda functions by evaluating strings.
The generator returns a function of npeaks number of profiles to be used for
curve fitting
The returned function is: returned_function(x, y0, h1, p1, w1, h2, p2, w2,
h3, p3, w3...)
"""


def lorentzians(npeaks=1):
    """Returns a function of npeaks number of lorentzian profiles"""

    height_labels = ["h{}".format(n) for n in range(1, npeaks + 1)]
    width_labels = ["w{}".format(n) for n in range(1, npeaks + 1)]
    position_labels = ["p{}".format(n) for n in range(1, npeaks + 1)]

    equation = "y0"
    expression = "lambda x,y0"

    for h, p, w in zip(height_labels, position_labels, width_labels):
        equation += "+ {0}/(((x-{1})**2)/({2}**2) + 1)".format(h, p, w)
        expression += ",{0},{1},{2}".format(h, p, w)

    return eval(expression + ": " + equation)


def gaussians(npeaks=1):
    """Returns a function of npeaks number of gaussian profiles"""

    height_labels = ["h{}".format(n) for n in range(1, npeaks + 1)]
    width_labels = ["w{}".format(n) for n in range(1, npeaks + 1)]
    position_labels = ["p{}".format(n) for n in range(1, npeaks + 1)]

    equation = "y0"
    expression = "lambda x,y0"

    for h, p, w in zip(height_labels, position_labels, width_labels):
        equation += "+ {0}*(2.71828183**(-0.69314718*((x-{1})/{2})**2))"\
                                                        .format(h, p, w)
        expression += ",{0},{1},{2}".format(h, p, w)

    return eval(expression + ": " + equation)


def pseudo_voight_50(npeaks=1):
    """Returns a function of npeaks number of pseudo-voight profiles with
    50% lorentzian/gaussian mixing"""
    height_labels = ["h{}".format(n) for n in range(1, npeaks + 1)]
    width_labels = ["w{}".format(n) for n in range(1, npeaks + 1)]
    position_labels = ["p{}".format(n) for n in range(1, npeaks + 1)]

    lor_mix = 0.5  # proportion og lorentzian. lor_mix = 1 - gauss_mix

    equation = "y0"
    expression = "lambda x,y0"

    for h, p, w in zip(height_labels, position_labels, width_labels):
        equation += "+ {0}*({3}*(2.71828183**(-0.69314718*((x-{1})/{2})**2))"\
                                                    .format(h, p, w, lor_mix)
        equation += " + (1-{0})/(((x-{1})**2)/({2}**2) + 1))".format(
            lor_mix, p, w
        )
        expression += ",{0},{1},{2}".format(h, p, w)

    return eval(expression + ": " + equation)
