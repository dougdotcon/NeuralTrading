"""
Neural Trading - Sistema de Trading com IA Neural
Pacote principal do sistema Neural Trading Cyberpunk Terminal
"""

__version__ = '1.0.0'
__author__ = 'AI Trading Team'

from .neural_config import (
    CYBERPUNK_COLORS,
    CYBERPUNK_SYMBOLS,
    POPULAR_ASSETS,
    TRADING_STRATEGIES,
    NEURAL_MODELS,
    RISK_SETTINGS,
    DEFAULT_PORTFOLIO,
    SYSTEM_CONFIG,
    get_timestamp,
    format_currency,
    format_percentage
)

__all__ = [
    'CYBERPUNK_COLORS',
    'CYBERPUNK_SYMBOLS',
    'POPULAR_ASSETS',
    'TRADING_STRATEGIES',
    'NEURAL_MODELS',
    'RISK_SETTINGS',
    'DEFAULT_PORTFOLIO',
    'SYSTEM_CONFIG',
    'get_timestamp',
    'format_currency',
    'format_percentage'
]

