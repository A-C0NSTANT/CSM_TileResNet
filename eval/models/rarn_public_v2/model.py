from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_ROOT_MODEL = Path(__file__).resolve().parents[3] / 'model.py'
_SPEC = spec_from_file_location('_root_history_model', _ROOT_MODEL)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


CNNModel = _MODULE.CNNModel
ResNetPolicyModel = _MODULE.ResNetPolicyModel
RankAwareResNetPolicyModel = _MODULE.RankAwareResNetPolicyModel
RankAwareResNetPublicPolicyModel = _MODULE.RankAwareResNetPublicPolicyModel
RankAwareResNetPublicV2PolicyModel = _MODULE.RankAwareResNetPublicV2PolicyModel
RankAwareResNetPublicV2LargePolicyModel = _MODULE.RankAwareResNetPublicV2LargePolicyModel
create_model = _MODULE.create_model
model_requires_public = _MODULE.model_requires_public
model_requires_history = _MODULE.model_requires_history
