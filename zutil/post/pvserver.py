from builtins import str
from builtins import range
from paraview.simple import *

import fabric
from invoke.context import Context
from fabric import Connection

from zutil import analysis

import logging

log = logging.getLogger("paramiko.transport")
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
log.addHandler(sh)

import multiprocessing as mp
from multiprocessing import Process, Value

process_id = None
use_multiprocess = True
# Uncomment for output logging
# logger = mp.get_logger()
# logger.addHandler(logging.StreamHandler(sys.stdout))
# logger.setLevel(mp.SUBDEBUG)


def pvserver(c, remote_dir, paraview_cmd, paraview_port, paraview_remote_port):

    with c.forward_remote(
        remote_port=int(paraview_remote_port), local_port=int(paraview_port)
    ), c.cd(remote_dir):
        # with cd(remote_dir):
        if not use_multiprocess:
            c.run("sleep 2;" + paraview_cmd + "</dev/null &>/dev/null&", pty=False)
        else:
            #    # run('sleep 2;'+paraview_cmd+'&>/dev/null',pty=False)
            c.run("sleep 2;" + paraview_cmd)  # , pty=False)
        # run(paraview_cmd+'</dev/null &>/dev/null',pty=False)
        # run('screen -d -m "yes"')
    # ssh asrc2 "(ls</dev/null &>/dev/null&) 2>&1; true" 2>/dev/null || echo
    # SSH connection or remote command failed - either of them returned
    # non-zero exit code $?


def pvcluster(
    c,
    remote_dir,
    paraview_home,
    paraview_args,
    paraview_port,
    paraview_remote_port,
    job_dict,
):

    with c.forward_remote(
        remote_port=int(paraview_remote_port), local_port=int(paraview_port)
    ):
        c.run("mkdir -p " + remote_dir)
        with c.cd(remote_dir):
            cmd_line = "mycluster --create pvserver.job --jobname=pvserver"
            cmd_line += " --jobqueue " + job_dict["job_queue"]
            cmd_line += " --ntasks " + job_dict["job_ntasks"]
            cmd_line += " --taskpernode " + job_dict["job_ntaskpernode"]
            if "vizstack" in paraview_args:
                cmd_line += " --script mycluster-viz-paraview.bsh"
            else:
                cmd_line += " --script mycluster-paraview.bsh"
            cmd_line += " --project " + job_dict["job_project"]
            c.run(
                cmd_line,
                env={"PARAVIEW_HOME": paraview_home, "PARAVIEW_ARGS": paraview_args},
            )
            c.run("chmod u+rx pvserver.job")
            c.run("mycluster --immediate --submit pvserver.job")


def port_test(c, rport, lport):
    # Run a test
    with c.forward_remote(remote_port=int(rport), local_port=int(lport)):
        c.run("cd", hide="everything")


def run_uname(c, with_tunnel):
    c.run("uname -a", hide="everything")


def test_ssh(status, **kwargs):
    _remote_host = analysis.data.data_host
    if "data_host" in kwargs:
        _remote_host = kwargs["data_host"]
    try:
        # env.use_ssh_config = True
        c = Connection(_remote_host)
        run_uname(c, False)
    except:
        status.value = 0
        return False
    return True


def test_ssh_mp(**kwargs):

    # print 'Starting test ssh'
    status = Value("i", 1)
    process_id = mp.Process(target=test_ssh, args=(status,), kwargs=kwargs)
    process_id.start()
    process_id.join()
    if status.value == 0:
        return False

    return True


def test_remote_tunnel(**kwargs):

    _remote_host = analysis.data.data_host
    if "data_host" in kwargs:
        _remote_host = kwargs["data_host"]

    try:
        # env.use_ssh_config = True
        c = Connection(_remote_host)
        run_uname(c, True)
    except:
        return False

    return True


def get_remote_port(**kwargs):

    _remote_host = analysis.data.data_host
    if "data_host" in kwargs:
        _remote_host = kwargs["data_host"]

    paraview_port = analysis.data.paraview_port
    if "paraview_port" in kwargs:
        paraview_port = kwargs["paraview_port"]

    paraview_remote_port = analysis.data.paraview_remote_port
    if "paraview_remote_port" in kwargs:
        paraview_remote_port = kwargs["paraview_remote_port"]
    else:
        # Attempt to find an unused remote port
        print("Attempting to find unused port in range 12000 to 13000")
        for p in range(12000, 13000):
            tp = Value("i", p)

            process_id = mp.Process(
                target=test_remote_port,
                args=(port_test, tp, paraview_port, _remote_host),
            )
            process_id.start()
            process_id.join()
            # print tp.value
            if tp.value != 0:
                break

        print("Selected Port: " + str(p))
        analysis.data.paraview_remote_port = p


def test_remote_port(port_test, port, paraview_port, remote_host):

    try:
        # env.use_ssh_config = True
        c = Connection(remote_host)
        port_test(c, port.value, paraview_port)
        return True
    except:
        port.value = 0
        return False


def pvserver_start(remote_host, remote_dir, paraview_cmd):
    if paraview_cmd is not None:
        # env.use_ssh_config = True
        c = Connection(remote_host)
        pvserver(c, remote_dir, paraview_cmd)


def pvserver_connect(**kwargs):
    """
    Be careful when adding to this function fabric execute calls do not play
    well with multiprocessing. Do not mix direct fabric execute call and
    mp based fabric execute calls
    """
    # global remote_data, data_dir, data_host, remote_server_auto
    # global paraview_cmd, process_id, paraview_port, paraview_remote_port
    # global process_id

    _paraview_cmd = analysis.data.paraview_cmd
    if "paraview_cmd" in kwargs:
        _paraview_cmd = kwargs["paraview_cmd"]

    if "-sp" in _paraview_cmd or "--client-host" in _paraview_cmd:
        print(
            "pvserver_process: Please only provide pvserver"
            "executable path and name without arguments"
        )
        print("e.g. mpiexec -n 1 /path_to_pvserver/bin/pvserver")
        return False

    # Add Check for passwordless ssh
    print("Testing passwordless ssh access")
    if not test_ssh_mp(**kwargs):
        print("ERROR: Passwordless ssh access to data host failed")
        return False
    print("-> Passed")

    # Add check for paraview version

    # Find free remote port
    get_remote_port(**kwargs)

    paraview_port = analysis.data.paraview_port
    if "paraview_port" in kwargs:
        paraview_port = kwargs["paraview_port"]

    if not use_multiprocess:
        pvserver_process(**kwargs)
    else:
        print("Starting pvserver connect")
        process_id = mp.Process(target=pvserver_process, kwargs=kwargs)
        process_id.start()
        # process_id.join()

    # time.sleep(6)

    ReverseConnect(paraview_port)

    return True


def pvcluster_process(**kwargs):
    pvserver_process(**kwargs)


def pvserver_process(**kwargs):

    # global remote_data, data_dir, data_host, remote_server_auto
    # global paraview_cmd, paraview_home, paraview_port, paraview_remote_port

    print("Starting pvserver process")

    _remote_dir = analysis.data.data_dir
    if "data_dir" in kwargs:
        _remote_dir = kwargs["data_dir"]
    _paraview_cmd = analysis.data.paraview_cmd
    if "paraview_cmd" in kwargs:
        _paraview_cmd = kwargs["paraview_cmd"]
    _paraview_home = analysis.data.paraview_home
    if "paraview_home" in kwargs:
        _paraview_home = kwargs["paraview_home"]
    paraview_port = analysis.data.paraview_port
    if "paraview_port" in kwargs:
        paraview_port = kwargs["paraview_port"]

    """
    _job_ntasks = 1
    if 'job_ntasks' in kwargs:
        _job_ntasks = kwargs['job_ntasks']
    """

    _remote_host = analysis.data.data_host
    if "data_host" in kwargs:
        _remote_host = kwargs["data_host"]

    # This global variable may have already been set so check
    paraview_remote_port = analysis.data.paraview_remote_port
    if "paraview_remote_port" in kwargs:
        paraview_remote_port = kwargs["paraview_remote_port"]
    else:
        # Attempt to find an unused remote port
        print("Attempting to find unused port in range 12000 to 13000")
        for p in range(12000, 13000):
            try:
                # env.use_ssh_config = True
                c = Connection(_remote_host)
                port_test(c, p, paraview_port)
                break
            except:
                pass
        print("Selected Port: " + str(p))
        analysis.data.paraview_remote_port = p

    if "job_queue" in kwargs:
        # Submit job

        remote_hostname = _remote_host[_remote_host.find("@") + 1 :]

        paraview_args = (
            " -rc --client-host="
            + remote_hostname
            + " -sp="
            + str(paraview_remote_port)
        )

        print(paraview_args)

        job_dict = {
            "job_queue": kwargs["job_queue"],
            "job_ntasks": kwargs["job_ntasks"],
            "job_ntaskpernode": kwargs["job_ntaskpernode"],
            "job_project": kwargs["job_project"],
        }
        if _paraview_home is not None:
            # env.use_ssh_config = True
            c = Connection(_remote_host)
            pvcluster(
                c,
                _remote_dir,
                _paraview_home,
                paraview_args,
                analysis.data.paraview_port,
                analysis.data.paraview_remote_port,
                job_dict,
            )
    else:
        # Run Paraview
        if "-sp" in _paraview_cmd or "--client-host" in _paraview_cmd:
            print(
                "pvserver_process: Please only provide pvserver"
                "executable path and name without arguments"
            )
            print("e.g. mpiexec -n 1 /path_to_pvserver/bin/pvserver")
            return False
        _paraview_cmd = (
            _paraview_cmd
            + " -rc --client-host=localhost -sp="
            + str(analysis.data.paraview_remote_port)
        )

        if _paraview_cmd is not None:
            # env.use_ssh_config = True
            c = Connection(_remote_host)
            pvserver(
                c,
                _remote_dir,
                _paraview_cmd,
                analysis.data.paraview_port,
                analysis.data.paraview_remote_port,
            )


def pvserver_disconnect():
    Disconnect()
    if process_id:
        process_id.terminate()
