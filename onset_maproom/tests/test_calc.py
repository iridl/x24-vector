import numpy as np
import pandas as pd
import xarray as xr

import calc


def test_dts():
    t = pd.date_range(start="2000-01-01", end="2005-02-28", freq="1D")
    # this is rr_mrg.sel(T=slice("2000", "2005-02-28")).isel(X=150, Y=150).precip
    values = [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.178533,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.076297,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.613994,
        4.145842,
        1.837427,
        0.162814,
        0.020677,
        0.0,
        1.765618,
        0.046113,
        0.0,
        0.0,
        0.0,
        0.075331,
        0.0,
        0.001392,
        0.314578,
        0.0,
        2.256729,
        9.710979,
        2.299267,
        1.386846,
        0.671914,
        0.550319,
        0.233897,
        6.049298,
        0.544117,
        0.238243,
        0.0,
        0.057973,
        1.502136,
        0.914677,
        9.971033,
        12.578513,
        4.084161,
        8.318914,
        9.434103,
        6.558233,
        4.150691,
        9.797739,
        1.823453,
        1.559392,
        1.525554,
        0.033275,
        0.005791,
        0.053375,
        0.037706,
        0.000376,
        0.0,
        0.046904,
        0.014851,
        0.0,
        0.0,
        0.0,
        0.0,
        0.013746,
        0.0,
        0.104771,
        0.550424,
        0.486443,
        0.202623,
        0.15565,
        0.088162,
        0.591434,
        0.484073,
        0.075584,
        0.0,
        1.550283,
        0.887993,
        2.172674,
        1.587737,
        3.479736,
        3.213629,
        2.175055,
        0.74553,
        6.379514,
        0.656978,
        0.159414,
        6.430022,
        0.439094,
        0.746011,
        0.28014,
        0.345991,
        0.065149,
        8.082086,
        0.14874,
        1.586067,
        1.009312,
        9.255059,
        8.280125,
        2.200297,
        5.011527,
        3.062533,
        6.342909,
        3.067153,
        9.136004,
        4.827989,
        6.755694,
        7.501081,
        15.779922,
        17.708204,
        2.365917,
        5.463901,
        12.58227,
        0.693505,
        1.125054,
        7.332282,
        6.062822,
        5.332923,
        6.468035,
        3.209999,
        4.309499,
        5.474716,
        9.741375,
        9.016099,
        3.963966,
        6.852372,
        3.46771,
        11.144149,
        3.083095,
        7.283878,
        1.878884,
        9.365307,
        3.557823,
        6.013811,
        0.91724,
        4.192626,
        2.328648,
        2.329957,
        4.16986,
        3.988161,
        5.27802,
        6.662162,
        3.611029,
        3.098602,
        5.336749,
        9.195411,
        4.223935,
        9.396879,
        3.990933,
        1.40205,
        5.196463,
        2.747482,
        5.74577,
        11.542407,
        6.128833,
        8.348987,
        10.49291,
        1.424523,
        7.91107,
        4.254604,
        3.153958,
        1.621415,
        0.597154,
        0.844243,
        2.504142,
        1.216158,
        7.934445,
        3.838415,
        0.81913,
        5.693459,
        8.180226,
        4.933387,
        1.788628,
        1.939759,
        1.343717,
        3.096729,
        4.014324,
        3.542946,
        3.279068,
        7.137427,
        0.396095,
        4.35338,
        2.483895,
        2.710342,
        5.416018,
        0.511533,
        0.428618,
        3.202748,
        0.256622,
        0.59926,
        0.991496,
        5.745539,
        1.747743,
        2.854228,
        4.077405,
        10.269378,
        0.652235,
        0.282629,
        2.408561,
        2.202169,
        0.092674,
        0.012446,
        0.049916,
        0.816946,
        0.293112,
        3.135521,
        0.074173,
        0.007658,
        0.001115,
        0.010211,
        0.8028,
        0.138378,
        0.0,
        0.32566,
        0.034442,
        0.214942,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.097826,
        1.005947,
        0.301894,
        0.0,
        0.186849,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.172356,
        10.754777,
        2.581911,
        0.131925,
        1.597011,
        0.197758,
        0.0,
        0.0,
        0.0,
        0.017734,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.059744,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.471799,
        0.68914,
        0.653878,
        1.303658,
        0.333286,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.910843,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.07266,
        0.142666,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.231145,
        0.0,
        0.072187,
        0.0,
        0.0,
        0.0,
        0.0,
        0.156108,
        3.839261,
        3.428691,
        0.941473,
        0.014079,
        0.0,
        0.030289,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.030015,
        0.715498,
        0.287746,
        1.676174,
        0.731081,
        3.525332,
        14.654359,
        13.009281,
        3.935042,
        5.829321,
        1.83136,
        0.139327,
        1.126296,
        1.16232,
        0.911264,
        16.782093,
        3.934745,
        0.022177,
        0.0,
        0.041964,
        9.890126,
        0.0,
        3.811365,
        0.507285,
        3.258365,
        4.307502,
        5.956409,
        16.817566,
        3.214268,
        1.077374,
        0.562197,
        0.0,
        0.00134,
        0.020259,
        0.015987,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.000958,
        0.017023,
        0.047402,
        0.405667,
        0.240281,
        0.0,
        0.0,
        0.0,
        1.06833,
        3.446288,
        0.082154,
        2.520098,
        0.322598,
        0.544076,
        0.0,
        0.344229,
        0.39248,
        1.392189,
        0.630458,
        4.852504,
        1.534202,
        0.746606,
        0.0,
        17.42565,
        2.806151,
        0.003314,
        0.584632,
        6.511308,
        7.531717,
        5.135433,
        2.470186,
        4.198995,
        3.978146,
        6.00894,
        0.059829,
        0.004766,
        0.0,
        0.0,
        0.0,
        0.0,
        0.217624,
        0.020746,
        0.065358,
        0.0,
        3.83088,
        0.973317,
        5.425416,
        5.561438,
        0.0,
        0.470905,
        2.60351,
        0.0,
        0.0,
        0.1758,
        1.811576,
        3.592968,
        11.917809,
        3.814522,
        3.144099,
        1.682752,
        4.00807,
        4.26886,
        0.933125,
        2.212816,
        3.129949,
        3.843049,
        16.19793,
        6.848283,
        8.732409,
        4.580019,
        2.50734,
        3.187433,
        3.464069,
        4.395224,
        6.963826,
        7.091905,
        4.653807,
        0.368812,
        3.51829,
        1.857014,
        14.10599,
        1.616931,
        2.5328,
        14.228808,
        9.237814,
        0.361939,
        0.888819,
        6.040326,
        7.721986,
        8.015229,
        7.341615,
        2.296613,
        11.926289,
        11.183476,
        3.055649,
        2.70754,
        10.424335,
        15.8988,
        8.938977,
        10.675817,
        0.025275,
        1.665411,
        7.205348,
        9.195422,
        7.849537,
        10.389763,
        12.834826,
        9.078284,
        4.950782,
        5.997156,
        11.911917,
        11.14992,
        4.190289,
        2.218117,
        2.855249,
        1.998433,
        1.394025,
        2.426401,
        3.373663,
        5.071145,
        1.310084,
        0.985362,
        0.367093,
        6.332088,
        3.005105,
        1.268393,
        0.321891,
        3.779885,
        6.025437,
        1.597916,
        5.85447,
        4.210143,
        2.261607,
        4.939619,
        0.039207,
        9.586336,
        0.645331,
        1.517511,
        1.099802,
        6.296143,
        4.617028,
        9.605053,
        5.974632,
        3.490519,
        0.025572,
        2.342289,
        1.264473,
        0.396759,
        0.506923,
        0.0,
        1.146431,
        0.691426,
        0.433476,
        0.563793,
        2.16346,
        3.948159,
        3.822077,
        2.563132,
        0.256942,
        0.0,
        0.117002,
        0.0,
        0.361211,
        0.000281,
        0.438437,
        0.923884,
        0.0,
        0.0,
        0.0,
        1.37263,
        1.238087,
        0.530963,
        1.859412,
        0.362692,
        0.0,
        0.0,
        0.007811,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02992,
        0.755414,
        0.0,
        0.007325,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.539445,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.355781,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.357352,
        0.0,
        0.0,
        0.081458,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.267171,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.009875,
        0.0,
        0.0,
        0.0,
        0.0,
        1.766152,
        3.232971,
        24.468853,
        4.874351,
        0.0,
        0.0,
        0.0,
        0.0,
        0.10842,
        0.0,
        0.0,
        9.835848,
        4.752406,
        1.334037,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.707404,
        0.06462,
        0.0,
        2.864181,
        5.174697,
        3.15517,
        0.089695,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.147204,
        4.835972,
        4.71407,
        21.24237,
        0.510804,
        0.0,
        0.0,
        6.640717,
        0.0,
        0.401247,
        1.701604,
        0.0,
        0.0,
        0.0,
        5.261262,
        0.875706,
        1.425467,
        0.0,
        0.0,
        0.544909,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.40167,
        0.628199,
        0.416751,
        0.0,
        0.0,
        0.851127,
        0.0,
        0.0,
        0.056165,
        16.81271,
        0.271342,
        0.0,
        0.0,
        0.0,
        2.946578,
        0.179771,
        0.045253,
        0.0,
        2.59189,
        1.218231,
        0.0,
        0.0,
        0.56799,
        1.594898,
        2.838348,
        0.253485,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.046116,
        0.035312,
        1.614071,
        3.921022,
        2.627032,
        1.088532,
        0.0,
        0.0,
        0.0,
        0.224956,
        0.55089,
        0.0,
        0.0,
        0.0,
        0.068536,
        0.90111,
        0.273575,
        0.158926,
        2.607463,
        0.609725,
        3.176702,
        2.162873,
        0.617285,
        3.406171,
        0.0,
        0.0,
        0.0,
        0.295912,
        0.0,
        0.0,
        0.101303,
        1.13359,
        1.277783,
        0.830277,
        2.699149,
        0.384051,
        0.355888,
        0.309386,
        3.162703,
        0.908184,
        1.582118,
        4.476885,
        0.051021,
        2.344054,
        8.50686,
        5.441457,
        1.995831,
        1.283507,
        7.443674,
        0.504622,
        1.151585,
        3.795001,
        8.09217,
        0.139953,
        0.216477,
        1.777782,
        1.741029,
        2.897305,
        1.341759,
        2.204526,
        5.090332,
        0.184779,
        2.092642,
        0.014578,
        2.354306,
        5.509358,
        1.969532,
        0.188864,
        1.077429,
        9.415661,
        15.889854,
        12.685659,
        1.093619,
        0.085397,
        4.006612,
        4.256728,
        0.0,
        0.216113,
        4.776238,
        9.66073,
        9.821564,
        4.841014,
        7.778699,
        4.731795,
        14.324765,
        7.96984,
        5.639346,
        5.578972,
        13.130465,
        13.927779,
        2.448434,
        0.846979,
        5.306067,
        11.231367,
        0.908376,
        4.578964,
        7.584468,
        1.606828,
        0.942186,
        0.572035,
        4.568054,
        1.811092,
        2.148642,
        1.636311,
        8.78365,
        8.156862,
        2.054801,
        4.308779,
        11.444415,
        6.397045,
        4.129268,
        1.548452,
        10.04043,
        9.031294,
        4.991867,
        5.779587,
        5.680329,
        4.176216,
        1.366309,
        0.61769,
        0.244638,
        0.602928,
        1.803758,
        1.159301,
        1.493993,
        0.144496,
        0.729026,
        0.000655,
        6.814487,
        3.221142,
        0.740212,
        0.178083,
        0.695517,
        1.111248,
        0.108652,
        1.810338,
        7.303304,
        2.779603,
        5.057292,
        0.785697,
        1.910726,
        0.957349,
        0.062848,
        0.0,
        0.130564,
        0.0,
        0.678137,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.006981,
        0.05219,
        0.0,
        0.0,
        0.0,
        0.950047,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.452691,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.179954,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.003321,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.440363,
        2.371839,
        0.393555,
        4.629946,
        0.492009,
        0.0,
        0.024091,
        0.0,
        0.019733,
        0.0,
        0.0,
        0.14865,
        0.165877,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00871,
        0.0,
        0.0,
        0.0,
        0.006964,
        0.0,
        0.0,
        1.038828,
        3.284928,
        2.673823,
        0.728778,
        0.0,
        0.0,
        0.0,
        0.0,
        0.549701,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.132058,
        0.0,
        0.0,
        0.0,
        0.866877,
        1.427235,
        6.583727,
        2.990242,
        0.238304,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.114329,
        0.0,
        0.164617,
        1.591755,
        27.02136,
        0.0,
        15.584168,
        3.096281,
        0.06766,
        0.0,
        0.0,
        0.0,
        0.122603,
        0.0,
        1.712193,
        4.85047,
        2.683853,
        2.521704,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        19.55956,
        3.284866,
        5.85433,
        4.431423,
        0.562957,
        1.957659,
        6.21776,
        2.627105,
        2.423535,
        2.90175,
        0.870144,
        5.81771,
        2.456054,
        5.010378,
        1.042031,
        0.426257,
        1.480479,
        2.779504,
        0.128967,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.562937,
        3.390016,
        0.89933,
        0.0,
        0.0,
        0.104532,
        3.571415,
        1.152433,
        0.332381,
        0.717392,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.323586,
        0.0,
        1.242237,
        0.378556,
        0.261257,
        1.689917,
        2.811209,
        0.383385,
        0.0,
        0.419859,
        0.23683,
        0.805206,
        0.963968,
        0.867919,
        10.252362,
        3.960352,
        8.1239,
        0.433884,
        2.034413,
        4.583262,
        5.24799,
        9.553278,
        3.633719,
        5.572575,
        4.087827,
        0.792047,
        7.951057,
        11.125514,
        4.936119,
        5.578837,
        0.965422,
        1.136402,
        15.808572,
        12.206754,
        0.865291,
        12.824273,
        10.09315,
        12.551981,
        2.024585,
        0.826361,
        22.90645,
        1.776212,
        0.514051,
        17.041683,
        7.682241,
        7.893931,
        2.445339,
        5.784092,
        17.104992,
        4.749784,
        10.638633,
        9.201806,
        0.0,
        19.931896,
        13.804424,
        9.839211,
        3.979795,
        0.633887,
        5.561112,
        3.739348,
        13.914742,
        8.635159,
        3.911247,
        15.883176,
        6.720156,
        1.744209,
        3.689742,
        1.88309,
        11.462723,
        6.831369,
        9.155698,
        6.284542,
        7.32334,
        2.395972,
        0.339289,
        0.37068,
        1.725673,
        5.443511,
        0.500955,
        0.583772,
        3.189042,
        10.773691,
        12.1475,
        5.978309,
        10.516581,
        9.887717,
        0.338566,
        7.616873,
        1.35397,
        10.254058,
        0.374095,
        3.404776,
        5.373506,
        3.083017,
        0.349521,
        1.402505,
        1.467141,
        2.411119,
        0.403345,
        6.761158,
        4.030739,
        8.632412,
        6.639418,
        0.924299,
        1.981852,
        4.325481,
        8.160402,
        3.135005,
        1.597651,
        3.110315,
        0.0,
        0.301568,
        4.289638,
        3.434969,
        1.768226,
        1.669349,
        0.819582,
        0.752483,
        1.525205,
        0.672672,
        1.218474,
        3.874151,
        5.167159,
        0.33789,
        0.001686,
        0.346527,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.429438,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.16181,
        0.0,
        0.0,
        0.0,
        0.196655,
        0.079196,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        16.066444,
        0.0,
        0.0,
        0.000407,
        0.0,
        0.0,
        0.0,
        0.556578,
        10.619227,
        0.837951,
        0.635848,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.1031,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.216507,
        0.235678,
        0.675946,
        11.65239,
        0.0,
        0.224911,
        0.001041,
        2.348371,
        1.937193,
        1.553942,
        0.042281,
        6.954834,
        0.0,
        0.056286,
        0.0,
        0.0,
        0.0,
        0.015718,
        3.504514,
        1.102076,
        1.287085,
        1.884331,
        0.555616,
        0.014926,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.004489,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.825379,
        0.0,
        0.423569,
        2.883357,
        1.267423,
        0.0,
        0.0,
        0.077122,
        0.032747,
        0.0,
        0.0,
        1.439739,
        1.535601,
        12.615751,
        0.304026,
        2.672402,
        1.035261,
        0.0,
        2.474638,
        0.0,
        0.066375,
        0.066042,
        1.053849,
        0.838401,
        2.620595,
        0.012447,
        7.283473,
        0.296261,
        0.0,
        0.077975,
        0.118308,
        23.931288,
        4.24107,
        5.627712,
        9.848227,
        2.361931,
        0.719009,
        13.53106,
        6.273913,
        1.82713,
        5.677546,
        16.816433,
        5.890654,
        0.634844,
        2.695535,
        1.108697,
        0.313671,
        3.418766,
        0.005782,
        0.0,
        0.0,
        0.276206,
        0.0,
        0.0,
        0.0,
        0.0,
        0.180438,
        0.822252,
        0.036085,
        3.570703,
        3.898225,
        2.063656,
        1.245327,
        0.309505,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.171351,
        0.236904,
        0.049628,
        0.707111,
        0.811659,
        1.732992,
        0.0,
        0.0,
        0.027575,
        0.55997,
        0.80942,
        1.919369,
        0.778496,
        1.331457,
        4.41746,
        4.754182,
        4.354914,
        0.495945,
        1.21819,
        1.423816,
        3.333878,
        4.165767,
        1.571782,
        9.667932,
        2.592115,
        3.8829,
        3.717976,
        0.033591,
        2.465083,
        0.990359,
        3.403143,
        16.761044,
        10.882872,
        2.372886,
        1.098031,
        6.587827,
        12.765206,
        3.485619,
        6.771199,
        4.030517,
        3.57429,
        5.64115,
        1.483279,
        0.532147,
        0.418907,
        2.363742,
        2.95837,
        0.938818,
        6.152651,
        13.766706,
        1.800182,
        3.730095,
        6.884064,
        5.870577,
        6.773752,
        2.396065,
        3.940956,
        3.53434,
        3.784236,
        9.422354,
        6.414804,
        1.778224,
        1.621394,
        3.969509,
        2.803879,
        3.808465,
        1.700168,
        1.764069,
        3.446691,
        8.128755,
        7.324597,
        3.041806,
        4.465253,
        5.945008,
        1.240393,
        10.673851,
        4.320079,
        4.180705,
        5.562878,
        3.057058,
        6.241708,
        0.653013,
        0.37586,
        1.401638,
        2.493164,
        0.197648,
        4.639569,
        3.70183,
        5.87538,
        0.792817,
        0.0,
        1.005437,
        6.440681,
        1.184689,
        1.788002,
        0.312534,
        2.061232,
        3.134323,
        2.057908,
        0.752729,
        1.912914,
        0.966592,
        1.160245,
        1.321875,
        0.934015,
        0.540155,
        0.85225,
        0.226253,
        2.289442,
        1.736535,
        4.226911,
        1.177538,
        0.964985,
        1.286437,
        2.70218,
        0.792248,
        1.2045,
        0.446366,
        0.516156,
        0.357902,
        0.389365,
        0.382511,
        1.626637,
        0.545314,
        0.0,
        0.9319,
        0.0,
        0.509477,
        0.120703,
        0.128327,
        2.925095,
        1.308689,
        0.34651,
        4.402626,
        0.482099,
        0.613916,
        1.956849,
        0.0,
        0.299069,
        0.169879,
        0.095796,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.130293,
        0.0,
        0.20588,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.154896,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.255222,
        0.126956,
        0.10043,
        0.006866,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.069079,
        0.0,
        0.0,
        0.0,
        0.0,
        0.001652,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.211269,
        5.749138,
        5.437428,
        3.361074,
        0.001031,
        0.085564,
        0.078808,
        2.277596,
        0.114164,
        0.0,
        0.0,
        0.0,
        0.0,
        4.02564,
        1.322553,
        0.0,
        0.2692,
        0.0,
        0.0,
        1.737253,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.451228,
        0.054314,
        0.333831,
        0.637705,
        6.406705,
    ]
    precip = xr.DataArray(values, dims=["T"], coords={"T": t}).rename("precip")

    dts = calc.daily_tobegroupedby_season(precip, 29, 11, 29, 2)

    assert dts["T"].size == 461
    assert dts["group"].size == 5
    assert (
        dts.seasons_ends
        == xr.DataArray(
            [
                "2001-02-28T00:00:00.000000000",
                "2002-02-28T00:00:00.000000000",
                "2003-02-28T00:00:00.000000000",
                "2004-02-29T00:00:00.000000000",
                "2005-02-28T00:00:00.000000000",
            ]
        )
    ).all

    onsetsds = calc.seasonal_onset_date(
        precip, 1, 3, 90, 1, 3, 20, 1, 7, 21, time_coord="T"
    )
    onsets = onsetsds.onset_delta + onsetsds["T"]
    assert (onsets == xr.DataArray([
            "NaT",
            "2001-03-08T00:00:00.000000000",
            "NaT",
            "2003-04-12T00:00:00.000000000",
            "2004-04-04T00:00:00.000000000",
        ])).all
   


def test_onset():

    t = pd.date_range(start="2000-05-01", end="2000-06-30", freq="1D")
    # this is rr_mrg.isel(X=0, Y=124, drop=True).sel(T=slice("2000-05-01", "2000-06-30"))
    values = [
        0.054383,
        0.0,
        0.0,
        0.027983,
        0.0,
        0.0,
        7.763758,
        3.27952,
        13.375934,
        4.271866,
        12.16503,
        9.706059,
        7.048605,
        0.0,
        0.0,
        0.0,
        0.872769,
        3.166048,
        0.117103,
        0.0,
        4.584551,
        0.787962,
        6.474878,
        0.0,
        0.0,
        2.834413,
        9.029134,
        0.0,
        0.269645,
        0.793965,
        0.0,
        0.0,
        0.0,
        0.191243,
        0.0,
        0.0,
        4.617332,
        1.748801,
        2.079067,
        2.046696,
        0.415886,
        0.264236,
        2.72206,
        1.153666,
        0.204292,
        0.0,
        5.239006,
        0.0,
        0.0,
        0.0,
        0.0,
        0.679325,
        2.525344,
        2.432472,
        10.737132,
        0.598827,
        0.87709,
        0.162611,
        18.794922,
        3.82739,
        2.72832,
    ]
    precip = xr.DataArray(values, dims=["T"], coords={"T": t})
    precipNaN = precip + np.nan

    onsets = calc.onset_date(
        daily_rain=precip,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
    )
    onsetsNaN = calc.onset_date(
        daily_rain=precipNaN,
        wet_thresh=1,
        wet_spell_length=3,
        wet_spell_thresh=20,
        min_wet_days=1,
        dry_spell_length=7,
        dry_spell_search=21,
    )
    assert pd.Timedelta(onsets.values) == pd.Timedelta(days=6)
    # Converting to pd.Timedelta doesn't change the meaning of the
    # assertion, but gives a more helpful error message when the test
    # fails: Timedelta('6 days 00:00:00')
    # vs. numpy.timedelta64(518400000000000,'ns')
    assert np.isnat(onsetsNaN.values)
