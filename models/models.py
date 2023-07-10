
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycleMorph':
        from .cycleMorph_model import cycleMorph
        model = cycleMorph()
    if opt.model == 'cycleregister':
        from .cycle_register_model import cycleregister
        model = cycleregister()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
