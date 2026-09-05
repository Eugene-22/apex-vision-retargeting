"""HumanHandState to MockApexHand safe end-to-end pipeline."""

from __future__ import annotations

from human_hand.state import HumanHandState
from robot.apex_urdf import ApexUrdfModel
from robot.interface import ApexCommandResult, ApexInterface
from robot.index_pipeline import map_index_to_full_target
from retargeting.calibration import IndexCalibration
from safety.command_chain import SafetyChainResult, SafetyCommandChain


class SafeIndexPipeline:
    """Map one human index state through safety into a backend."""

    def __init__(self, model: ApexUrdfModel, calibration: IndexCalibration, hand: ApexInterface, chain: SafetyCommandChain) -> None:
        self._model = model
        self._calibration = calibration
        self._hand = hand
        self._chain = chain

    def process(self, state: HumanHandState | None) -> tuple[SafetyChainResult, ApexCommandResult] | None:
        previous = self._hand.get_joint_state().position
        if state is None:
            return None
        raw = map_index_to_full_target(state.index_angles, self._calibration, self._model, base=previous)
        result = self._chain.process_stateful(raw, state.timestamp, confidence=state.quality_score)
        command = self._hand.command_position(result.target)
        return result, command
