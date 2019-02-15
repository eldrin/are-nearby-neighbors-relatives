from collections import OrderedDict


class Config:
#     PERTURBATIONS = OrderedDict([
#         ('PS', [-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12]),  # 14
#         ('TS', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]),  # 14
#         ('PN', [-15, -10, -6, -3, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),  # 12
#         ('EN', [-15, -10, -6, -3, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]),  # 12
#         ('MP', [8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128, 160, 192]),  # 13
#     ])
    PERTURBATIONS = OrderedDict([
        ('PS', [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]),  # 8
        ('TS', [0.9, 0.925, 0.95, 0.975, 1.025, 1.05, 1.075, 1.1]),  # 8
        ('PN', [23, 24, 25, 26, 27, 28, 29, 30]),  # 8
        ('EN', [23, 24, 25, 26, 27, 28, 29, 30]),  # 8
        ('MP', [80, 96, 128, 160, 192, 224, 256, 320]),  # 6
    ])