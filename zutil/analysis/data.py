"""
Copyright (c) 2012-2024, Zenotech Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Zenotech Ltd nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ZENOTECH LTD BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os

remote_data = False
remote_server_auto = True

data_dir = None
ref_data_dir = None
data_host = None
paraview_cmd = None

paraview_remote_port = "11113"
paraview_port = "11111"


def init(
    default_data_dir: None = None,
    default_data_host: None = None,
    default_paraview_cmd: str = "mpiexec.hydra pvserver",
    default_ref_data_dir: None = None,
) -> None:
    """Initialise directories"""

    global data_dir, ref_data_dir, data_host, paraview_cmd

    if "DATA_DIR" in os.environ:
        data_dir = os.environ["DATA_DIR"]
    else:
        data_dir = default_data_dir

    if "REF_DATA_DIR" in os.environ:
        ref_data_dir = os.environ["REF_DATA_DIR"]
    else:
        if default_ref_data_dir is None:
            ref_data_dir = data_dir
        else:
            ref_data_dir = default_ref_data_dir

    if "PARAVIEW_CMD" in os.environ:
        paraview_cmd = os.environ["PARAVIEW_CMD"]
    else:
        paraview_cmd = default_paraview_cmd

    if not remote_server_auto:
        paraview_cmd = None

    if not remote_data:
        data_host = "localhost"
        paraview_cmd = None
