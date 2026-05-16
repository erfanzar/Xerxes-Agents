# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional Weights & Biases logging hook for RL training runs.

Acts as a no-op when ``wandb`` is not installed or ``WANDB_API_KEY``
isn't set, so calling code can stay unconditional.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class WandBHook:
    """Forward RL metrics to Weights & Biases when available."""

    def __init__(self, project: str = "xerxes-rl", entity: str | None = None) -> None:
        """Bind to a WandB ``project`` and optional ``entity`` (org/team)."""
        self._project = project
        self._entity = entity
        self._run: Any | None = None

    def is_available(self) -> bool:
        """Return ``True`` when both ``wandb`` and ``WANDB_API_KEY`` exist."""
        return importlib.util.find_spec("wandb") is not None and bool(os.environ.get("WANDB_API_KEY"))

    def start(self, config: dict[str, Any], *, name: str | None = None) -> str:
        """Open a WandB run and return its dashboard URL (``""`` if unavailable)."""
        if not self.is_available():
            logger.debug("WandB unavailable; hook is a no-op")
            return ""
        import wandb  # type: ignore

        run = wandb.init(project=self._project, entity=self._entity, name=name, config=config, reinit=True)
        self._run = run
        return getattr(run, "url", "")

    def log(self, metrics: dict[str, Any]) -> None:
        """Log a metrics dict against the active run (no-op when none)."""
        if self._run is None:
            return
        self._run.log(metrics)

    def finish(self) -> None:
        """Close the active WandB run and clear the reference."""
        if self._run is None:
            return
        try:
            self._run.finish()
        finally:
            self._run = None


__all__ = ["WandBHook"]
