import numpy as np
import tensorflow as tf

import locale

locale.setlocale(locale.LC_ALL, '')

_params = {}
def param(name, *args, **kwargs):
    """
    A wrapper for `tf.Variable` which enables parameter sharing in models.

    Creates and returns theano shared variables similarly to `tf.Variable`,
    except if you try to create a param with the same name as a
    previously-created one, `param(...)` will just return the old one instead of
    making a new one.
    This constructor also adds a `param` attribute to the shared variables it
    creates, so that you can easily search a graph for all params.
    """

    if name not in _params:
        kwargs['name'] = name
        param = tf.Variable(*args, **kwargs)
        param.param = True
        _params[name] = param
    return _params[name]

def params_with_name(name):
    return [p for n,p in _params.items() if name in n]

def delete_all_params():
    _params.clear()

def print_model_settings(locals_):
    print "Model settings:"
    all_vars = [(k,v) for (k,v) in locals_.items() if (k.isupper() and k!='T')]
    all_vars = sorted(all_vars, key=lambda x: x[0])
    for var_name, var_value in all_vars:
        print "\t{}: {}".format(var_name, var_value)
