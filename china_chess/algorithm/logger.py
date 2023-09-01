from enum import Enum
from file_writer import write_line


class LogType(Enum):
    INFO = 0
    DEBUG = 1
    ERROR = 2
    WARNING = 3
    DISPLAY = 4


type2name = {
    LogType.INFO: "INFO",
    LogType.DEBUG: "DEBUG",
    LogType.ERROR: "ERROR",
    LogType.WARNING: "WARNING",
    LogType.DISPLAY: "DISPLAY",
}

type2fore = {
    LogType.INFO: "35",
    LogType.DEBUG: "34",
    LogType.ERROR: "31",
    LogType.WARNING: "33",
    LogType.DISPLAY: "36",
}
time2event = {}


def log(log_type, msg, is_in_file=False, file_name='logger.txt'):
    print("\033[1;" + type2fore[log_type] + ";40m{:15s}".format(type2name[log_type]) + msg)
    if is_in_file:
        write_line(file_name, msg)


if __name__ == '__main__':
    log(LogType.DISPLAY, "fff", is_in_file=True)
    log(LogType.DISPLAY, "fff", is_in_file=True)
