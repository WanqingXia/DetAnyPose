def choose_from_candidate(img, pred, dinov2):
    # more than one candidate mask and need to choose best one
    if len(pred['labels']) > 1:
        stop =1
    else:
        raise ValueError("More than one candidate masks is required by choose_from_candidate")


def choose_from_viewpoints(img, pred, dinov2):
    # choose the best viewpoint from the mask
    if len(pred['labels']) == 1:
        stop = 1
    else:
        raise ValueError("Only one mask can be processed with choose_from_viewpoints")