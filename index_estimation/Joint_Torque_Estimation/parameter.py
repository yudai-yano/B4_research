KEYPOINT_NAMES = [
    (0, 'nose'),
    (1, 'neck'),
    (2, 'right_shoulder'),
    (3, 'right_elbow'),
    (4, 'right_hand'),
    (5, 'left_shoulder'),
    (6, 'left_elbow'),
    (7, 'left_hand'),
    (8, 'right_waist'),
    (9, 'right_knee'),
    (10, 'right_ankle'),
    (11, 'left_waist'),
    (12, 'left_knee'),
    (13, 'left_ankle'),
    (14, 'right_eye'),
    (15, 'left_eye'),
    (16, 'right_ear'),
    (17, 'left_ear')
]

KEYPOINT_IDS = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

AE_PARTMASS_EST_A0_MALE = {
    'head': -1.1968,
    'body': -10.1647,
    'upper_arm': -0.36785,
    'fore_arm': -0.43807,
    'hand': -0.01474,
    'thigh': -4.53542,
    'crus': -1.71524,
    'foot': -0.26784,
    'upper_torso': -9.63322,
    'lower_torso': -5.70449
}

AE_PARTMASS_EST_A1_MALE = {
    'head': 25.9526,
    'body': 18.7503,
    'upper_arm': 1.15588,
    'fore_arm': 2.22923,
    'hand': 2.09424,
    'thigh': 14.5253,
    'crus': 6.04396,
    'foot': 2.61804,
    'upper_torso': 31.431,
    'lower_torso': 29.2113
}

AE_PARTMASS_EST_A2_MALE = {
    'head': 0.02604,
    'body': 0.48275,
    'upper_arm': 0.02712,
    'fore_arm': 0.01397,
    'hand': 0.00414,
    'thigh': 0.09324,
    'crus': 0.03885,
    'foot': 0.00545,
    'upper_torso': 0.27893,
    'lower_torso': 0.17966
}

AE_PARTMASS_EST_A0_FEMALE = {
    'head': -0.67895,
    'body': -14.61,
    'upper_arm': -0.49429,
    'fore_arm': -0.33838,
    'hand': -0.04586,
    'thigh': -3.74942,
    'crus': -1.40459,
    'foot': -0.39879,
    'upper_torso': -6.14877,
    'lower_torso': -6.87637
}

AE_PARTMASS_EST_A1_FEMALE = {
    'head': 22.8598,
    'body': 34.646,
    'upper_arm': 2.04431,
    'fore_arm': 2.42059,
    'hand': 1.68034,
    'thigh': 8.19775,
    'crus': 5.11653,
    'foot': 3.20138,
    'upper_torso': 26.8538,
    'lower_torso': 39.0758
}

AE_PARTMASS_EST_A2_FEMALE = {
    'head': 0.02111,
    'body': 0.39995,
    'upper_arm': 0.02414,
    'fore_arm': 0.01079,
    'hand': 0.00478,
    'thigh': 0.13576,
    'crus': 0.04346,
    'foot': 0.00477,
    'upper_torso': 0.22507,
    'lower_torso': 0.17224
}

AE_COM_RATIO_MALE = {
    'head': 0.821,
    'body': 0.493,
    'upper_arm': 0.529,
    'fore_arm': 0.415,
    'hand': 0.891,
    'thigh': 0.475,
    'crus': 0.406,
    'foot': 0.595,
    'upper_torso': 0.428,
    'lower_torso': 0.609
}

AE_COM_RATIO_FEMALE = {
    'head': 0.759,
    'body': 0.506,
    'upper_arm': 0.523,
    'fore_arm': 0.423,
    'hand': 0.908,
    'thigh': 0.458,
    'crus': 0.410,
    'foot': 0.594,
    'upper_torso': 0.438,
    'lower_torso': 0.597
}

ABE_PARTMASS_EST_A0_MALE = {
    'head': 1.521,
    'body': -71.133,
    'upper_arm': -3.378,
    'fore_arm': -1.454,
    'hand': -0.097,
    'thigh': -13.261,
    'crus': -4.757,
    'foot': -0.843,
    'upper_torso': -45.839,
    'lower_torso': -2.153
}

ABE_PARTMASS_EST_A1_MALE = {
    'head': 0,
    'body': 0,
    'upper_arm': 0,
    'fore_arm': -0.003,
    'hand': -0.001,
    'thigh': -0.016,
    'crus': 0,
    'foot': 0,
    'upper_torso': 0.041,
    'lower_torso': -0.087
}

ABE_PARTMASS_EST_A2_MALE = {
    'head': 0.034,
    'body': 0,
    'upper_arm': 0.007,
    'fore_arm': 0.004,
    'hand': 0.006,
    'thigh': 0.0041,
    'crus': 0.015,
    'foot': 0.005,
    'upper_torso': 0,
    'lower_torso': 0.213
}

ABE_PARTMASS_EST_A3_MALE = {
    'head': -0.032,
    'body': 0,
    'upper_arm': 0,
    'fore_arm': 0,
    'hand': -0.007,
    'thigh': 0,
    'crus': 0,
    'foot': 0,
    'upper_torso': 0.121,
    'lower_torso': 0.069
}

ABE_PARTMASS_EST_A4_MALE = {
    'head': 0.131,
    'body': 0.399,
    'upper_arm': 0.077,
    'fore_arm': 0.040,
    'hand': 0.025,
    'thigh': 0.224,
    'crus': 0.056,
    'foot': 0.055,
    'upper_torso': 0.427,
    'lower_torso': 0.489
}

ABE_PARTMASS_EST_A5_MALE = {
    'head': 0,
    'body': 0.389,
    'upper_arm': 0.098,
    'fore_arm': 0.070,
    'hand': 0,
    'thigh': 0.245,
    'crus': 0.126,
    'foot': 0,
    'upper_torso': 0.436,
    'lower_torso': 0
}
ABE_PARTMASS_EST_A6_MALE = {
    'head': 0,
    'body': 0.305,
    'upper_arm': 0,
    'fore_arm': 0,
    'hand': 0,
    'thigh': 0,
    'crus': 0,
    'foot': 0,
    'upper_torso': 0,
    'lower_torso': 0
}

ABE_PARTMASS_EST_A7_MALE = {
    'head': 0,
    'body': 0.213,
    'upper_arm': 0,
    'fore_arm': 0,
    'hand': 0,
    'thigh': 0,
    'crus': 0,
    'foot': 0,
    'upper_torso': 0,
    'lower_torso': 0
}

ABE_PARTMASS_EST_A0_FEMALE = {
    'head': 1.224,
    'body': -42.904,
    'upper_arm': -2.550,
    'fore_arm': -0.698,
    'hand': 0.099,
    'thigh': -11.452,
    'crus': -3.592,
    'foot': -0.526,
    'upper_torso': -6.848,
    'lower_torso': -18.275
}

ABE_PARTMASS_EST_A1_FEMALE = {
    'head': 0,
    'body': -0.068,
    'upper_arm': 0,
    'fore_arm': -0.004,
    'hand': -0.001,
    'thigh': 0,
    'crus': -0.012,
    'foot': 0,
    'upper_torso': -0.050,
    'lower_torso': -0.050
}

ABE_PARTMASS_EST_A2_FEMALE = {
    'head': 0.027,
    'body': 0,
    'upper_arm': 0,
    'fore_arm': 0.007,
    'hand': 0.005,
    'thigh': 0.036,
    'crus': 0.016,
    'foot': 0.007,
    'upper_torso': 0.192,
    'lower_torso': 0
}

ABE_PARTMASS_EST_A3_FEMALE = {
    'head': -0.013,
    'body': 0,
    'upper_arm': 0,
    'fore_arm': -0.003,
    'hand': -0.003,
    'thigh': 0,
    'crus': 0,
    'foot': -0.004,
    'upper_torso': 0,
    'lower_torso': 0
}

ABE_PARTMASS_EST_A4_FEMALE = {
    'head': 0.166,
    'body': 0.576,
    'upper_arm': 0.055,
    'fore_arm': 0.030,
    'hand': 0.018,
    'thigh': 0.174,
    'crus': 0.079,
    'foot': 0.039,
    'upper_torso': 0.298,
    'lower_torso': 0.546
}

ABE_PARTMASS_EST_A5_FEMALE = {
    'head': 0,
    'body': 0.170,
    'upper_arm': 0.105,
    'fore_arm': 0.051,
    'hand': 0,
    'thigh': 0.207,
    'crus': 0.125,
    'foot': 0,
    'upper_torso': 0.099,
    'lower_torso': 0
}
ABE_PARTMASS_EST_A6_FEMALE = {
    'head': 0,
    'body': 0.475,
    'upper_arm': 0,
    'fore_arm': 0,
    'hand': 0,
    'thigh': 0,
    'crus': 0,
    'foot': 0,
    'upper_torso': 0,
    'lower_torso': 0.228
}

ABE_PARTMASS_EST_A7_FEMALE = {
    'head': 0,
    'body': 0,
    'upper_arm': 0,
    'fore_arm': 0,
    'hand': 0,
    'thigh': 0,
    'crus': 0,
    'foot': 0,
    'upper_torso': 0,
    'lower_torso': 0.108
}

ABE_COM_RATIO_MALE_A1 = {
    'head': 0.821,
    'body': 0.493,
    'upper_arm': 0.529,
    'fore_arm': 0.415,
    'hand': 0.891,
    'thigh': 0.475,
    'crus': 0.406,
    'foot': 0.595,
    'upper_torso': 0.428,
    'lower_torso': 0.609
}
