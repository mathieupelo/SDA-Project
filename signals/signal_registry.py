from typing import Dict, List
from signals.signal_base import SignalBase


class SignalRegistry:
    """Registry to manage all available signals"""

    def __init__(self):
        self._signals: Dict[str, SignalBase] = {}

    def register(self, signal: SignalBase):
        """Register a new signal"""
        self._signals[signal.name] = signal

    def get_signal(self, name: str) -> SignalBase:
        """Get signal by name"""
        return self._signals.get(name)

    def list_signals(self) -> List[str]:
        """List all available signal names"""
        return list(self._signals.keys())
