from nn_trust.factory import Factory, Info
from nn_trust.loss._loss import Loss


class InfoLoss(Info[Loss]):
    pass


class LossFactory(Factory):
    """
    Register-Factory for storing the losses
    """
    _info_type = InfoLoss
