"""
Send signals to the running experiment
For example when you'd like the training loop to pause and manually debug the training process
And resume training without any pain
"""
import signal
import psutil
from easyvolcap.utils.console_utils import *


def find_most_parent_process(name: str):
    # Iterate over all processes currently running on the system
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
        try:
            # Convert process name and command line to lower case for case insensitive matching
            cmdline = ' '.join(proc.cmdline()).lower() if proc.cmdline() else ''
            process_name = proc.name().lower()
            if 'python' not in cmdline or 'evc-sig' in cmdline or 'sigusr1.py' in cmdline:
                continue

            # Check if the process name or command line contains the specified name
            if name.lower() in cmdline or name.lower() in process_name:
                # Check the parent process
                parent = proc.parent()
                if parent:
                    # Convert parent process name and command line to lower case for case insensitive matching
                    parent_cmdline = ' '.join(parent.cmdline()).lower() if parent.cmdline() else ''
                    parent_name = parent.name().lower()

                    if name.lower() in parent_cmdline or name.lower() in parent_name:
                        # Check the parent process
                        gparent = parent.parent()
                        if gparent:
                            # Convert gparent process name and command line to lower case for case insensitive matching
                            gparent_cmdline = ' '.join(gparent.cmdline()).lower() if gparent.cmdline() else ''
                            gparent_name = gparent.name().lower()

                            # If gparent does not match the specified name, return this process
                            if name.lower() in gparent_cmdline or name.lower() in gparent_name:
                                # Check the parent process
                                ggparent = gparent.parent()
                                if ggparent:
                                    # Convert ggparent process name and command line to lower case for case insensitive matching
                                    ggparent_cmdline = ' '.join(ggparent.cmdline()).lower() if ggparent.cmdline() else ''
                                    ggparent_name = ggparent.name().lower()

                                    if name.lower() not in ggparent_cmdline and name.lower() not in ggparent_name:
                                        return proc.pid  # Return the PID of this process

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # The process may have ended or may not be accessible due to permissions during scanning
            continue

    return None  # Return None if no matching most parent process is found


def send_signal(name: str, signal: int = signal.SIGUSR1):
    pid = find_most_parent_process(name)
    if pid is not None:
        log(f'Sending signal {green(signal)} to {green(name)} with PID {pid}')
        os.kill(pid, signal)
    else:
        log(red(f'Unable to find any top-level parent process with the specified name: {yellow(name)}'))


@catch_throw
def main():
    # Assuming build_parser and dotdict are properly defined and implemented elsewhere in your code
    args = dotdict(
        name='tfgs_flame_salmon',
        signal='SIGUSR1' if os.name != 'nt' else 'SIGBREAK',
    )
    args = dotdict(vars(build_parser(args, description=__doc__).parse_args()))
    if isinstance(args.signal, str):
        args.signal = getattr(signal, args.signal)
    send_signal(args.name, args.signal)


if __name__ == '__main__':
    main()
