
import os
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from configparser import RawConfigParser

### CG-RBI modules ###
from CGrbi.envtools.path import create_path

def init_logging(project_name : str, dir_pref : Path, config : RawConfigParser) -> logging.Logger:
    
    log_dir = create_path(os.path.join(dir_pref, config.get('Log','DirLog')))
        
    log_file_name_debug = config.get('Log', 'BaseLogFileNameForDebug')
    path_log_file_name_debug = os.path.join(log_dir, log_file_name_debug)
    
    log_file_name_info = config.get('Log', 'BaseLogFileNameInfo')
    path_log_file_name_info = os.path.join(log_dir, log_file_name_info)
    
    logger = logging.getLogger(project_name+"_logger")
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(config.get('Log','LogFormat'))
    file_handler_debug = RotatingFileHandler(path_log_file_name_debug, 
                                             mode='a',
                                            maxBytes=config.getint('Log', 'MaxBytes'),
                                            backupCount=1)
    file_handler_debug.setLevel(logging.DEBUG)
    file_handler_debug.setFormatter(formatter)
    
    fileHandler = RotatingFileHandler(path_log_file_name_info, mode='a',
                                      maxBytes=config.getint('Log', 'MaxBytes'),
                                      backupCount=1)
    fileHandler.setLevel(logging.INFO)
    fileHandler.setFormatter(formatter)
    
    logger.addHandler(file_handler_debug)
    logger.addHandler(fileHandler)
    
    return logger