from nn_trust.attack.evasion.fom import FOMAttackConfig
from services.utils.utils import get_parameter_prop


def test_parameter_metadata_preserves_zero_defaults():
    fields = FOMAttackConfig.model_fields

    assert get_parameter_prop("momentum", fields["momentum"]).default == 0.0
    assert get_parameter_prop("dampening", fields["dampening"]).default == 0.0

