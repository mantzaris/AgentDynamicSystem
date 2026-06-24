import json
import re
import subprocess
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np

from agent_dynamic_system.benchmark import MetricSpec


class CodexGenericPolicy:
    """Codex-in-the-loop policy for generic benchmark systems."""

    def __init__(
        self,
        system_name: str,
        strategy_name: str,
        action_type: Type[Any],
        action_limits: Dict[str, float],
        metric_spec: MetricSpec,
        fallback_policy: Callable[[Any, Any], Any],
        decision_interval: int = 20,
        timeout_seconds: int = 120,
        codex_command: str = "codex",
        guardian_threshold: float = 0.04,
    ) -> None:
        self.system_name = system_name
        self.strategy_name = strategy_name
        self.action_type = action_type
        self.action_limits = action_limits
        self.metric_spec = metric_spec
        self.fallback_policy = fallback_policy
        self.decision_interval = max(1, int(decision_interval))
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.codex_command = codex_command
        self.guardian_threshold = guardian_threshold
        self._history: List[Dict[str, float]] = []
        self._session_id: Optional[str] = None

    def __call__(self, observation: Any, config: Any) -> Any:
        self._record_history(observation)
        if not self._should_consult():
            return self.action_type()

        advisory = None
        if self.strategy_name == "codex_control_advised":
            advisory = self._control_advisory(observation)

        prompt = self._build_prompt(observation, config, advisory)
        try:
            response = self._ask_codex(prompt)
        except Exception:
            return self.fallback_policy(observation, config)
        return self._parse_action(response, observation, config)

    def _should_consult(self) -> bool:
        step = int(self._history[-1]["step"])
        if self.strategy_name in ("codex_steady", "codex_control_advised"):
            return step % self.decision_interval == 0
        if self.strategy_name == "codex_guardian":
            if step % max(1, self.decision_interval // 4) != 0:
                return False
            return self._guardian_triggered()
        return False

    def _guardian_triggered(self) -> bool:
        if len(self._history) < 4:
            return False
        scores = [self._state_instability(item) for item in self._history[-6:]]
        recent = float(np.mean(scores[-2:]))
        previous = float(np.mean(scores[:-2]))
        if recent - previous > self.guardian_threshold:
            return True
        latest = self._history[-1]
        for name, upper in self.metric_spec.safety_max.items():
            if name in latest and latest[name] > 0.85 * upper:
                return True
        for name, lower in self.metric_spec.safety_min.items():
            if name in latest and latest[name] < 1.15 * lower:
                return True
        return False

    def _record_history(self, observation: Any) -> None:
        payload = asdict(observation)
        self._history.append({key: float(value) for key, value in payload.items()})
        self._history = self._history[-16:]

    def _state_instability(self, state: Dict[str, float]) -> float:
        scores = []
        for name, target in self.metric_spec.target.items():
            if name in state:
                scores.append(abs(state[name] - target) / max(abs(target), 1.0e-9))
        return float(np.mean(scores) if scores else 0.0)

    def _control_advisory(self, observation: Any) -> Dict[str, Any]:
        state = asdict(observation)
        errors = {}
        for name, target in self.metric_spec.target.items():
            if name in state:
                errors[name] = {
                    "target": target,
                    "current": state[name],
                    "normalized_error": (state[name] - target) / max(abs(target), 1.0e-9),
                }

        recommended_action = "none"
        recommended_amount = 0.0
        largest_error_name = None
        largest_error = 0.0
        for name, error in errors.items():
            magnitude = abs(float(error["normalized_error"]))
            if magnitude > largest_error:
                largest_error = magnitude
                largest_error_name = name

        if self.action_limits and largest_error > 0.15:
            action_names = list(self.action_limits)
            index = abs(hash((self.system_name, largest_error_name))) % len(action_names)
            recommended_action = action_names[index]
            recommended_amount = min(
                self.action_limits[recommended_action],
                max(0.02, 0.18 * largest_error),
            )

        return {
            "method": "simple proportional target-error advisory",
            "errors": errors,
            "recommended_action": recommended_action,
            "recommended_amount": recommended_amount,
            "lower_is_better": True,
        }

    def _build_prompt(
        self,
        observation: Any,
        config: Any,
        advisory: Optional[Dict[str, Any]],
    ) -> str:
        state = {
            "system": self.system_name,
            "strategy": self.strategy_name,
            "decision_interval": self.decision_interval,
            "current_observation": asdict(observation),
            "recent_history": self._history,
            "targets": self.metric_spec.target,
            "safety_min": self.metric_spec.safety_min,
            "safety_max": self.metric_spec.safety_max,
            "allowed_actions": {
                "none": 0.0,
                **self.action_limits,
            },
            "control_theory_advisory": advisory,
            "config": _compact_config(config),
        }
        trigger = (
            "You were called because recent instability appears to be worsening.\n\n"
            if self.strategy_name == "codex_guardian"
            else ""
        )
        return (
            f"{trigger}"
            "You are a Codex agent acting as an intervention controller inside "
            "a dynamic-system simulation. Choose exactly one intervention for "
            "the current step. The action is applied once. Optimize the "
            "lower-is-better instability metric: stay near targets, avoid safety "
            "bounds, reduce sharp changes, and avoid unnecessary action cost.\n\n"
            "Return only JSON with this shape:\n"
            "{\"action\":\"none|ACTION_NAME\",\"amount\":0.0,\"reason\":\"short reason\"}\n\n"
            f"Simulation payload:\n{json.dumps(state, indent=2, sort_keys=True)}"
        )

    def _ask_codex(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(mode="r", suffix=".txt", delete=False) as output_file:
            output_path = Path(output_file.name)
        command = self._codex_command(output_path)
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                check=True,
            )
            self._remember_session_id(completed.stdout)
            return output_path.read_text().strip()
        finally:
            try:
                output_path.unlink()
            except OSError:
                pass

    def _codex_command(self, output_path: Path) -> List[str]:
        if self._session_id is None:
            return [
                self.codex_command,
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--json",
                "--output-last-message",
                str(output_path),
                "--color",
                "never",
                "-",
            ]
        return [
            self.codex_command,
            "exec",
            "resume",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            str(output_path),
            self._session_id,
            "-",
        ]

    def _remember_session_id(self, stdout: str) -> None:
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = _extract_session_id(event)
            if session_id is not None:
                self._session_id = session_id

    def _parse_action(self, response: str, observation: Any, config: Any) -> Any:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return self.fallback_policy(observation, config)
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return self.fallback_policy(observation, config)
        action = str(payload.get("action", "none")).strip()
        try:
            amount = float(payload.get("amount", 0.0))
        except (TypeError, ValueError):
            amount = 0.0
        if action not in self.action_limits or amount <= 0.0:
            return self.action_type()
        amount = min(max(amount, 0.0), self.action_limits[action])
        return self.action_type(**{action: amount})


def _compact_config(config: Any) -> Dict[str, Any]:
    payload = {}
    for item in fields(config):
        value = getattr(config, item.name)
        if isinstance(value, (int, float, str, bool)):
            payload[item.name] = value
    return payload


def _extract_session_id(value: Any) -> Optional[str]:
    uuid_pattern = (
        r"[0-9a-fA-F]{8}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{12}"
    )
    if isinstance(value, dict):
        for key, item in value.items():
            key_lower = str(key).lower()
            if isinstance(item, str) and (
                "session" in key_lower or "conversation" in key_lower or "thread" in key_lower
            ):
                if re.fullmatch(uuid_pattern, item.strip()):
                    return item.strip()
            nested = _extract_session_id(item)
            if nested is not None:
                return nested
    if isinstance(value, list):
        for item in value:
            nested = _extract_session_id(item)
            if nested is not None:
                return nested
    return None
