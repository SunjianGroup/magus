import logging


c_formatter = logging.Formatter("%(asctime)s   %(message)s", datefmt='%H:%M:%S')
f_formatter = logging.Formatter("%(message)s")
f_formatter_para = logging.Formatter('[%(asctime)s PID:%(process)d %(levelname)s] %(message)s', datefmt='%H:%M:%S')

log_level = {
    "DEBUG":   logging.DEBUG, 
    "INFO":    logging.INFO, 
    "WARNING": logging.WARNING, 
    "ERROR":   logging.ERROR}


def set_logger(name=None, level="INFO", log_path="log.txt", formatter_para = False):
    level = log_level[level]
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(c_formatter)
    ch.setLevel(logging.DEBUG)
    log.addHandler(ch)
    fh = logging.FileHandler(log_path)
    if formatter_para:
        fh.setFormatter(f_formatter_para)
    else:
        fh.setFormatter(f_formatter)
    fh.setLevel(level)
    log.addHandler(fh)
