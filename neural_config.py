#!/usr/bin/env python3
"""
🔥 NEURAL TRADING CONFIG 🔥
Configurações e constantes para o sistema NeuralTrading
"""

import os
from datetime import datetime

# Cores cyberpunk
CYBERPUNK_COLORS = {
    'primary': '\033[96m',      # Ciano
    'secondary': '\033[92m',    # Verde
    'accent': '\033[93m',       # Amarelo
    'danger': '\033[91m',       # Vermelho
    'info': '\033[94m',         # Azul
    'warning': '\033[95m',      # Magenta
    'reset': '\033[0m'          # Reset
}

# Símbolos cyberpunk
CYBERPUNK_SYMBOLS = {
    'arrow': '►',
    'bullet': '●',
    'diamond': '◆',
    'square': '■',
    'triangle': '▲',
    'circle': '○',
    'star': '★',
    'lightning': '⚡',
    'gear': '⚙',
    'chart': '📈',
    'money': '💰',
    'robot': '🤖',
    'fire': '🔥',
    'rocket': '🚀'
}

# Configurações de animação
ANIMATION_SPEED = 0.1
LOADING_FRAMES = ['▓', '▒', '░', '▒']

# Assets populares para trading
POPULAR_ASSETS = {
    'stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'],
    'crypto': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'AVAX', 'MATIC'],
    'forex': ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF'],
    'commodities': ['GOLD', 'SILVER', 'OIL', 'COPPER', 'WHEAT', 'CORN']
}

# Estratégias de trading disponíveis
TRADING_STRATEGIES = {
    'momentum': {
        'name': 'Momentum Trading',
        'description': 'Segue tendências de alta/baixa com sinais neurais',
        'risk_level': 'Medium',
        'timeframe': '1h-4h',
        'sharpe_target': 2.84
    },
    'mean_reversion': {
        'name': 'Mean Reversion',
        'description': 'Arbitragem estatística com ML',
        'risk_level': 'Low',
        'timeframe': '15m-1h',
        'sharpe_target': 2.90
    },
    'swing': {
        'name': 'Swing Trading',
        'description': 'Análise multi-timeframe com sentimento',
        'risk_level': 'Medium',
        'timeframe': '4h-1d',
        'sharpe_target': 1.89
    },
    'mirror': {
        'name': 'Mirror Trading',
        'description': 'Copia estratégias institucionais',
        'risk_level': 'High',
        'timeframe': '1d-1w',
        'sharpe_target': 6.01
    }
}

# Modelos neurais disponíveis
NEURAL_MODELS = {
    'nhits': {
        'name': 'NHITS',
        'description': 'Neural Hierarchical Interpolation for Time Series',
        'accuracy': '94.7%',
        'latency': '2.3ms',
        'gpu_speedup': '6250x'
    },
    'nbeats': {
        'name': 'N-BEATS',
        'description': 'Neural Basis Expansion Analysis',
        'accuracy': '92.1%',
        'latency': '3.1ms',
        'gpu_speedup': '4890x'
    },
    'tft': {
        'name': 'TFT',
        'description': 'Temporal Fusion Transformer',
        'accuracy': '91.8%',
        'latency': '4.7ms',
        'gpu_speedup': '3200x'
    }
}

# Configurações de risco
RISK_SETTINGS = {
    'conservative': {
        'max_position_size': 0.02,  # 2% por posição
        'max_portfolio_risk': 0.10,  # 10% do portfólio
        'stop_loss': 0.05,  # 5% stop loss
        'take_profit': 0.15  # 15% take profit
    },
    'moderate': {
        'max_position_size': 0.05,  # 5% por posição
        'max_portfolio_risk': 0.20,  # 20% do portfólio
        'stop_loss': 0.08,  # 8% stop loss
        'take_profit': 0.25  # 25% take profit
    },
    'aggressive': {
        'max_position_size': 0.10,  # 10% por posição
        'max_portfolio_risk': 0.40,  # 40% do portfólio
        'stop_loss': 0.12,  # 12% stop loss
        'take_profit': 0.40  # 40% take profit
    }
}

# Configurações de portfólio padrão
DEFAULT_PORTFOLIO = {
    'initial_capital': 100000.0,  # $100k inicial
    'currency': 'USD',
    'risk_profile': 'moderate',
    'rebalance_frequency': 'weekly'
}

# Timeframes disponíveis
TIMEFRAMES = {
    '1m': '1 minute',
    '5m': '5 minutes',
    '15m': '15 minutes',
    '1h': '1 hour',
    '4h': '4 hours',
    '1d': '1 day',
    '1w': '1 week'
}

# Indicadores técnicos
TECHNICAL_INDICATORS = [
    'RSI', 'MACD', 'Bollinger Bands', 'Moving Averages',
    'Stochastic', 'Williams %R', 'CCI', 'ADX'
]

def ensure_directories():
    """Garante que os diretórios necessários existem"""
    directories = [
        'data',
        'models',
        'results',
        'logs',
        'backtest_results'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Diretório criado: {directory}")

def get_timestamp():
    """Retorna timestamp formatado"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def format_currency(value, currency='USD'):
    """Formata valor monetário"""
    if currency == 'USD':
        return f"${value:,.2f}"
    elif currency == 'BTC':
        return f"₿{value:.8f}"
    elif currency == 'EUR':
        return f"€{value:,.2f}"
    else:
        return f"{value:,.2f} {currency}"

def format_percentage(value):
    """Formata porcentagem"""
    return f"{value:.2f}%"

def get_risk_color(risk_level):
    """Retorna cor baseada no nível de risco"""
    if risk_level.lower() == 'low':
        return CYBERPUNK_COLORS['secondary']  # Verde
    elif risk_level.lower() == 'medium':
        return CYBERPUNK_COLORS['accent']     # Amarelo
    elif risk_level.lower() == 'high':
        return CYBERPUNK_COLORS['danger']     # Vermelho
    else:
        return CYBERPUNK_COLORS['info']       # Azul

# Configurações de sistema
SYSTEM_CONFIG = {
    'version': '1.0.0',
    'name': 'NeuralTrading MVP',
    'description': 'AI-Powered Neural Trading Platform',
    'author': 'AI Trading Team',
    'gpu_enabled': True,
    'max_concurrent_strategies': 4,
    'data_refresh_interval': 60,  # segundos
    'log_level': 'INFO'
}

# Mensagens do sistema
SYSTEM_MESSAGES = {
    'welcome': 'Bem-vindo ao NeuralTrading - Sistema de Trading com IA',
    'startup': 'Inicializando sistema neural...',
    'shutdown': 'Desconectando do sistema...',
    'error': 'Erro no sistema detectado',
    'success': 'Operação concluída com sucesso',
    'warning': 'Atenção: Verifique os parâmetros',
    'info': 'Informação do sistema'
}
