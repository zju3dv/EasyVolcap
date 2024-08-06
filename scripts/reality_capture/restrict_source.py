import psutil
import os
import time
rc_pid = None
while True:
    new_rc_pid = None
    pid_list = psutil.pids()
    for pid in pid_list:
        try:
            p = psutil.Process(pid)
            if "RealityCapture" in p.name():
                new_rc_pid = pid
                if True:
                    rc_pid = new_rc_pid
                    print(f'wmic process where "processid={rc_pid}" call setpriority "idle"')
                    os.system(f'wmic process where "processid={rc_pid}" call setpriority "idle"')
                    p.cpu_affinity([0, 1, 2, 3])

                    # rc_pid = new_rc_pid
                    # print(f'wmic process where "processid={rc_pid}" call setpriority "normal"'1)
                    # os.system(f'wmic process where "processid={rc_pid}" call setpriority "normal"')
                    # p.cpu_affinity([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
                else:
                    break
        except:
            continue
    # print(new_rc_pid, rc_pid)
    # if new_rc_pid and new_rc_pid != rc_pid:
    #     rc_pid = new_rc_pid
    #     print(f'wmic process where "processid={rc_pid}" call setpriority "idle"')
    #     # print(f'wmic process where "processid={rc_pid}" CALL setaffinity 1,2')
    #     os.system(f'wmic process where "processid={rc_pid}" call setpriority "idle"')
    #     # os.system(f'wmic process where "processid={rc_pid}" CALL setaffinity 1,2')
    #     p.cpu_affinity([0,1,2,3])
    else:
        time.sleep(2)
