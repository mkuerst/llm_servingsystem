from my_project.common.basemodel.infer_struct import InferStateInfo
from my_project.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from my_project.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight


class BaseLayerInfer:
    def __init__(self) -> None:
        pass

    def context_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def token_forward(self, input_ids, infer_state: InferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")

    def splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: BaseLayerWeight):
        raise Exception("need to impl")
