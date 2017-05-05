import os
import configparser


def load_parameters(parameters_filepath=os.path.join('.','parameters.ini'), verbose=True):
    '''
    Load parameters from the ini file, and ensure that each parameter is cast to the correct type
    '''
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath)
    nested_parameters = convert_configparser_to_dictionary(conf_parameters)
    parameters = {}
    for k, v in nested_parameters.items():
        parameters.update(v)
    for k, v in parameters.items():
        # If the value is a list delimited with a comma, choose one element at random.
        # Ensure that each parameter is cast to the correct type
        if k in {'n_hidden_1', 'image_size', 'number_of_classes', 'batch_size', 'training_epochs',
                 'n_hidden_2', 'display_step'}:
            parameters[k] = int(v)
        elif k in ['learning_rate']:
            parameters[k] = float(v)

    return parameters


def convert_configparser_to_dictionary(config):
    '''
    http://stackoverflow.com/questions/1773793/convert-configparser-items-to-dictionary
    '''
    my_config_parser_dict = {s:dict(config.items(s)) for s in config.sections()}
    return my_config_parser_dict
