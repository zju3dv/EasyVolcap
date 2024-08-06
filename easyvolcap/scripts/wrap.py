import os
import sys
import subprocess

from easyvolcap.utils.console_utils import *

easyvolcap = f'{dirname(__file__)}/../..'
EASYVOLCAP = f'python -q -X faulthandler {easyvolcap}/easyvolcap/scripts/main.py'


def configurable_entrypoint(SEPERATION='--', LAUNCHER='', EASYVOLCAP=EASYVOLCAP,
                            default_launcher_args=[],
                            extra_launcher_args=[],
                            default_easyvolcap_args=[],
                            extra_easyvolcap_args=[],
                            ):
    # Prepare for args
    args = sys.argv
    if SEPERATION in args:
        launcher_args = args[1:args.index(SEPERATION)]
        easyvolcap_args = args[args.index(SEPERATION) + 1:]
    else:
        launcher_args = default_launcher_args  # no extra arguments for torchrun (auto communimation, all available gpus)
        easyvolcap_args = args[1:] if len(args[1:]) else default_easyvolcap_args
    launcher_args += extra_launcher_args
    easyvolcap_args += extra_easyvolcap_args

    # Prepare for invokation
    args = []
    args.append(LAUNCHER)
    if launcher_args: args.append(' '.join(launcher_args))
    args.append(EASYVOLCAP)
    if easyvolcap_args: args.append(' '.join(easyvolcap_args))

    # The actual invokation
    subprocess.call(' '.join(args), shell=True)


def dist_entrypoint():
    # Distribuated training
    configurable_entrypoint(LAUNCHER='torchrun', EASYVOLCAP=f'{easyvolcap}/easyvolcap/scripts/main.py', default_launcher_args=['--nproc_per_node', 'auto'], extra_easyvolcap_args=['distributed=True'])


def prof_entrypoint():
    # Profiling
    if not [s for s in sys.argv if 'runner_cfg.epochs' in s]:
        sys.argv.append('runner_cfg.epochs=1')

    if not [s for s in sys.argv if 'runner_cfg.ep_iter' in s]:
        sys.argv.append('runner_cfg.ep_iter=50')

    if not [s for s in sys.argv if 'runner_cfg.eval_ep' in s]:
        sys.argv.append('runner_cfg.eval_ep=50')

    configurable_entrypoint(extra_easyvolcap_args=['profiler_cfg.enabled=True'])


def test_entrypoint():
    configurable_entrypoint(EASYVOLCAP=EASYVOLCAP + ' ' + '-t test')


def train_entrypoint():
    configurable_entrypoint(EASYVOLCAP=EASYVOLCAP + ' ' + '-t train')


def main_entrypoint():
    configurable_entrypoint()


def gui_entrypoint():
    # Directly run GUI without external requirements
    if '-c' not in sys.argv:
        sys.argv.insert(1, '-c')
        sys.argv.insert(2, 'configs/specs/gui.yaml')

    configurable_entrypoint(EASYVOLCAP=EASYVOLCAP + ' ' + '-t gui')


def ws_entrypoint():
    # Directly run GUI without external requirements
    if '-c' not in sys.argv:
        sys.argv.insert(1, '-c')
        sys.argv.insert(2, 'configs/base.yaml')

    args = sys.argv
    args = [f'python -q -X faulthandler {easyvolcap}/easyvolcap/scripts/client.py'] + args[1:]
    subprocess.call(' '.join(args), shell=True)


def sig_entrypoint():
    args = sys.argv
    args = [f'python -q -X faulthandler {easyvolcap}/easyvolcap/scripts/sigusr1.py'] + args[1:]
    subprocess.call(' '.join(args), shell=True)
