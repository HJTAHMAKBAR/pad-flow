def print_progress_log(epoch: int, logs: dict, extra=None):
    console_print = f'\x1b[2K\rEpoch {epoch:3}:'
    console_print += ''.join(f" [{key}]{value:5.3f}" for key, value in logs.items())

    if extra is not None:
        if isinstance(extra, str):
            console_print += '| ' + extra
        elif isinstance(extra, list) and len(extra) > 0:
            console_print += '  | ' + "".join(f' {info}' for info in extra)

    print(console_print)