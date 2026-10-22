"""Frozen pre-consolidation source outputs, independent of the runtime kernels.

Captured on CPU from looped-networks revision
3ee575baeceb1ad784102162a051b7444182fdaa, via the untouched original
neuro-autoresearch checkout. Synthetic inputs only; no neural datasets.
"""
import numpy as np
import pytest
import torch

from neuro_eval import encoding

SOURCE_BASELINES = {'tall': {'alphas': [37.92690190732238, 37.92690190732238, 10000000000.0],
          'scores': [-0.13028845191001892, -0.5260267853736877, -0.3330216705799103],
          'prediction': [[0.022279167547822, -0.03515550494194031, -1.7399376384119591e-10],
                         [-0.007729372475296259, -0.006772255524992943, -9.240379295061274e-11],
                         [-0.029164496809244156, 0.027308909222483635, -1.1110844883033266e-11],
                         [0.05189934000372887, -0.023657727986574173, -2.2143276101616038e-10],
                         [0.07191529124975204, -0.04273929074406624, 2.519787989374578e-10],
                         [-0.028180105611681938, -0.048032406717538834, -5.04135566625763e-10],
                         [0.005674810614436865, 0.042802125215530396, -2.1550686235549676e-10]]},
 'wide': {'alphas': [3.359818286283774, 3.359818286283774, 3.359818286283774],
          'scores': [-0.09804017096757889, -0.2485073059797287, -0.4164963364601135],
          'prediction': [[-0.00874994695186615, -0.0838826447725296, -0.2854132354259491],
                         [0.07739073038101196, 0.1769963651895523, 0.19625821709632874],
                         [0.1841927468776703, -0.06033969670534134, 0.2119370996952057],
                         [-0.021149002015590668, -0.11484994739294052, 0.01730358973145485],
                         [-0.1015128418803215, 0.019214199855923653, 0.24623499810695648],
                         [0.07547199726104736, 0.035152778029441833, 0.10708735883235931],
                         [0.013556532561779022, -0.3424701988697052, 0.09875260293483734]]},
 'rank_deficient': {'alphas': [37.92690190732238, 6951927.961775591, 10000000000.0],
                    'scores': [-0.23567646741867065, -0.4419967532157898, -0.3658258616924286],
                    'prediction': [[0.02152606099843979,
                                    -1.4222945310393698e-07,
                                    -1.3299694678892138e-10],
                                   [-0.010911082848906517,
                                    -4.247523222034033e-09,
                                    -3.6575843048725076e-11],
                                   [-0.023938970640301704,
                                    9.221023589134347e-08,
                                    -2.2930430124135093e-11],
                                   [0.06210155040025711,
                                    -5.126157276436061e-08,
                                    -1.8200688178815483e-10],
                                   [0.06300804018974304,
                                    -1.6477366671097116e-07,
                                    1.7932316742630405e-10],
                                   [-0.02792321890592575,
                                    -2.6249358597851824e-07,
                                    -3.976823037543653e-10],
                                   [0.02866854891180992,
                                    2.3168533402895264e-07,
                                    -1.8424993475374407e-10]]},
 'ties': {'alphas': [10000000000.0, 10000000000.0, 10000000000.0],
          'scores': [0.0, 0.0, 0.0],
          'prediction': [[0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0]]}}


@pytest.mark.parametrize("case,width", [("tall", 4), ("wide", 25),
                                        ("rank_deficient", 4), ("ties", 4)])
def test_encoding_frozen_source_regression(case, width):
    rng = np.random.RandomState(173)
    X = torch.tensor(rng.normal(size=(23, width)), dtype=torch.float32)
    Y = torch.tensor(rng.normal(size=(23, 3)), dtype=torch.float32)
    if case == "rank_deficient":
        X = torch.cat([X, X[:, :2], torch.zeros((23, 1))], 1)
    if case == "ties":
        Y[:] = 0
    tr, te = np.arange(16), np.arange(16, 23)
    Z = encoding.design(X, tr, "cpu", features="center")
    alphas, actual, prediction = encoding._fit_predict(Z, Y, tr, te)
    baseline = SOURCE_BASELINES[case]
    np.testing.assert_allclose(alphas, baseline["alphas"], rtol=1e-14)
    np.testing.assert_allclose(prediction, baseline["prediction"], rtol=2e-5, atol=1e-14)
    np.testing.assert_allclose(encoding.pearson(actual, prediction), baseline["scores"],
                               rtol=2e-5, atol=1e-6)
