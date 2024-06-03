""" -------------- project configulations difinition ---------------- """
# keys = config_name
# value = config_value

data_processing = {
    'input': rf'C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\input_data\NN_input_result.csv',
    'output': rf'C:\Users\yota0\Desktop\kamiya\program_copy\NeuralNetwork\result',
    'notuse': ['rest', 'time'],
    }

'''
lac_pred = {
    'columns': ['velocity','waist', 'knee', 'peak_time', 'lac', 'RM'],
    'explonatory_variable': ['velocity','waist', 'knee', 'peak_time'],
    'objective_variable': ['lac'],

    'train_row': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26],
    'test_row': [15 ,16, 17],
    'build_model': {
        'nodes': [256, 128, 64],
        'activation': ['relu', 'relu', 'relu', 'linear'],
        'optimizer': 'adamax',
        'loss': 'mse',
        'metrics': 'mae'
    },
    'model_fit': {
        'batch_size': 8,
        'epochs': 1000,
        'validation_split': 0.1,
    }
}, 'RM', 'RM', 'rest_time', 'rest_time'
'''
lac_pred = {
    'columns': ['velocity','waist', 'knee', 'peak_time', 'rest_time', 'lac', 'RM'],
    'explonatory_variable': ['velocity','waist', 'knee', 'peak_time', 'rest_time'],
    'objective_variable': ['lac'],

    'train_row': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26],
    'test_row': [15 ,16, 17],
    'build_model': {
        'nodes': [512 , 256 , 64 , 32],
        'activation': ['relu', 'relu', 'relu', 'linear'],
        'optimizer': 'adamax',
        'loss': 'mse',
        'metrics': 'mae'
    },
    'model_fit': {
        'batch_size': 8,
        'epochs': 5000,
        'validation_split': 0.1,
    }
}

'''
lac_pred = {
    'columns': ['velocity','waist', 'knee', 'peak_time', 'lac', 'RM'],
    'explonatory_variable': ['velocity','waist', 'knee', 'peak_time'],
    'objective_variable': ['lac'],

    'train_row': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26],
    'test_row': [15 ,16, 17],
    'build_model': {
        'nodes': [512 , 256 , 64 , 32],
        'activation': ['relu', 'relu', 'relu', 'linear'],
        'optimizer': 'adamax',
        'loss': 'mse',
        'metrics': 'mae'
    },
    'model_fit': {
        'batch_size': 8,
        'epochs': 1000,
        'validation_split': 0.1,
    }
}
'''

emg_pred = {
    'explonatory_variable': ['velocity','waist', 'knee',  'peak_time'],
    # 'objective_variable': ['rectus_femoris', 'vastus_lateralis', 'vastus_medialis', 'biceps_femoris'],
    'objective_variable': ['vastus_medialis'],
    'train_row': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    'test_row': [15 ,16, 17],
    'notuse': ['rest', 'time'],
    'build_model': {
        'nodes': [512, 256, 128, 32],
        'activation': ['relu', 'relu', 'relu', 'relu', 'linear'],
        'optimizer': 'adamax',
        'loss': 'mse',
        'metrics': 'mae'
    },
    'model_fit': {
        'batch_size': 1,
        'epochs': 1000,
        'validation_split': 0.1,
    }
}

fatigue_pred = {
    'explonatory_variable': ['velocity','waist', 'knee', 'peak_time'],
    'objective_variable': ['fatigue'],
    'notuse': ['rest', 'time'],
    'train_row': [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16, 17, 18, 19, 20, 21, 22, 23,],
    'test_row': [3, 4, 5],
    'build_model': {
        'nodes': [256, 128, 64, 32],
        'units': 4,
        'activation': ['relu', 'relu', 'relu', 'relu', 'softmax'],
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy',
        'metrics': 'accuracy'
    },
    'model_fit': {
        'batch_size': 4,
        'epochs': 1000,
        'validation_split': 0.1,
    }
}