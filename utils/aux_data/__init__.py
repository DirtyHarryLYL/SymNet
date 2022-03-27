import importlib

def load_loss_weight(dataset_name):
    """Loss weight to balance the categories
    weight = -log(frequency)"""

    if dataset_name[-1]=='g':
        dataset_name = dataset_name[:-1]
    
    try:
        Weight = importlib.import_module('utils.aux_data.%s_weight'%dataset_name)
        
        if 'pair_weight' in Weight.__dict__:
            return Weight.attr_weight, Weight.obj_weight, Weight.pair_weight
        else:
            return Weight.attr_weight, Weight.obj_weight, None

    except ImportError:
        raise NotImplementedError("Loss weight for %s is not implemented yet"%dataset_name)


def load_wordvec_dict(dataset_name, vec_type):
    if dataset_name[-1]=='g':
        dataset_name = dataset_name[:-1]
    try:
        Wordvec = importlib.import_module('utils.aux_data.%s_%s'%(vec_type, dataset_name))
    except ImportError:
        raise NotImplementedError("%s vector for %s is not ready yet"%(vec_type, dataset_name))


    if hasattr(Wordvec,'objs_dict'):
        return Wordvec.attrs_dict, Wordvec.objs_dict
    else:
        return Wordvec.attrs_dict