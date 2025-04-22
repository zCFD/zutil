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

import collections
import yaml
import os


def invalid_definition(definition: str) -> None:
    print("Invalid definition")
    print(definition)
    raise


# Dictionary of validation cases
validation_dict = collections.OrderedDict()

# Dictionary of test cases
case_dict = collections.OrderedDict()


def get_case(solve_params: dict, case_dict: dict) -> None:
    """Extracts solver parameters into case_dict"""
    for params in solve_params:
        if isinstance(params["controldict"], str):
            controldict = os.path.basename(params["controldict"])
            # Remove .py extension
            controldict = controldict.rsplit(".", 1)[0]
            case_dict[controldict] = params
            if "solve" in params:
                get_case(params["solve"], case_dict)
        else:
            import copy

            mesh_list = params["meshname"]
            case_list = params["controldict"]

            for m, c in zip(mesh_list, case_list):
                controldict = os.path.basename(c)
                # Remove .py extension
                controldict = controldict.rsplit(".", 1)[0]
                params1 = copy.deepcopy(params)
                params1["meshname"] = m
                params1["controldict"] = c
                case_dict[controldict] = params1
                if "solve" in params:
                    get_case(params["solve"], case_dict)


def get_case_dict(default_test_file: str = "generic.zcfd-test.yml") -> None:
    if "TEST_DEF" in os.environ:
        test_file = os.environ["TEST_DEF"]
    else:
        test_file = default_test_file

    # Get case names from yml file
    test_definition = None
    with open(test_file, "r") as test_definition_file:
        try:
            test_definition = yaml.safe_load(test_definition_file)
        except yaml.YAMLError as exc:
            print(exc)

    if test_definition is None:
        invalid_definition(test_definition)

    for step in test_definition:
        if len(list(step.keys())) > 1:
            invalid_definition(test_definition)

        for key in step:
            if key == "solve":
                get_case(step[key], case_dict)
            if key == "validation":
                for params in step[key]:
                    case = params["case"]
                    validation_dict[case] = params
