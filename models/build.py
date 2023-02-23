from utils import registry


MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a model, defined by `model_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Model: a constructed model specified by mdoel_name.
    """
    return MODELS.build(cfg, **kwargs)


