"""Deserialize a string using literal_eval

This is more than just literal_eval in that strings don't need quotes
and true/false are mapped to True/False

"""

from __future__ import print_function
import ast

class PyEvalError(Exception):
    message = ""
    def __init__(self, message=""):
        self.message = message
        super(PyEvalError, self).__init__(message)


class _Transformer(ast.NodeTransformer):
    """Turn a string into python objects (strings, numbers, as basic types)"""
    def visit_Name(self, node):
        if node.id.lower() in ["true", "false"]:
            # turn 'true' / 'True' / 'TRUE' / etc. into True
            val = True if node.id.lower() == "true" else False
            try:
                node = ast.copy_location(ast.NameConstant(val), node)
            except AttributeError:
                node = ast.copy_location(ast.Name(str(val), node.ctx), node)
        elif node.id.lower() == "none":
            val = None
            try:
                node = ast.copy_location(ast.NameConstant(val), node)
            except AttributeError:
                node = ast.copy_location(ast.Name(str(val), node.ctx), node)
        else:
            # turn other bare names into strings
            node = ast.copy_location(ast.Str(s=node.id), node)
        return self.generic_visit(node)


def parse(s):
    try:
        tree = ast.parse(s.strip(), mode='eval')
        _Transformer().visit(tree)
        return ast.literal_eval(tree)
    except ValueError:
        raise PyEvalError("ast parser vomited")

# def parse_option_list(s):
#     tree = ast.parse(s.strip(), mode='eval')
#     _Transformer().visit(tree)
#     return ast.literal_eval(tree)

# if __name__ == "__main__":
#     _s = """dict(opt1, opt2=12, opt3=text, opt4='some string')"""
#     print(parse_option_list(_s))

##
## EOF
##
