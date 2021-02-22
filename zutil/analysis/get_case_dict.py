import collections
import yaml
import os
import sys


def invalid_definition(definition):
    print("Invalid definition")
    print(definition)
    raise


# Dictionary of validation cases
validation_dict = collections.OrderedDict()

# Dictionary of test cases
case_dict = collections.OrderedDict()


def get_case(solve_params, case_dict):
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
            mesh_list = params['meshname']
            case_list = params['controldict']

            for m, c in zip(mesh_list, case_list):
                controldict = os.path.basename(c)
                # Remove .py extension
                controldict = controldict.rsplit(".", 1)[0]
                params1 = copy.deepcopy(params)
                params1['meshname'] = m
                params1['controldict'] = c
                case_dict[controldict] = params1
                if "solve" in params:
                    get_case(params["solve"], case_dict)


def get_case_dict(default_test_file="generic.zcfd-test.yml"):

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
