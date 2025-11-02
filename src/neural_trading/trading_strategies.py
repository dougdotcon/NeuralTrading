#!/usr/bin/env python3
"""
🔥 TRADING STRATEGIES ENGINE 🔥
Estratégias de trading com IA para o sistema NeuralTrading
Implementa Momentum, Mean Reversion, Swing e Mirror Trading
"""

import random
import time
import numpy as np
from datetime import datetime, timedelta
from .neural_config import TRADING_STRATEGIES, RISK_SETTINGS, get_timestamp, format_percentage
from .neural_forecaster import NeuralForecaster

class TradingStrategy:
    """Classe base para estratégias de trading"""
    
    def __init__(self, strategy_type, risk_profile='moderate'):
        self.strategy_type = strategy_type
        self.risk_profile = risk_profile
        self.strategy_info = TRADING_STRATEGIES.get(strategy_type)
        self.risk_settings = RISK_SETTINGS.get(risk_profile)
        self.forecaster = NeuralForecaster()
        self.active_positions = {}
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
    def analyze_market(self, symbol):
        """Análise de mercado base"""
        # Obtém previsão neural
        prediction = self.forecaster.predict(symbol, horizon=24)
        signal_strength = self.forecaster.get_signal_strength(prediction)
        trend_analysis = self.forecaster.analyze_trend(symbol)
        
        return {
            'symbol': symbol,
            'prediction': prediction,
            'signal': signal_strength,
            'trend': trend_analysis,
            'timestamp': get_timestamp()
        }
    
    def calculate_position_size(self, capital, risk_per_trade=None):
        """Calcula tamanho da posição baseado no risco"""
        if risk_per_trade is None:
            risk_per_trade = self.risk_settings['max_position_size']
        
        position_size = capital * risk_per_trade
        return position_size
    
    def generate_signal(self, market_analysis):
        """Gera sinal de trading - implementado pelas subclasses"""
        raise NotImplementedError("Subclasses devem implementar generate_signal")
    
    def execute_trade(self, signal, capital):
        """Executa trade baseado no sinal"""
        if signal['action'] == 'HOLD':
            return None
            
        position_size = self.calculate_position_size(capital)
        
        trade = {
            'symbol': signal['symbol'],
            'action': signal['action'],
            'price': signal['entry_price'],
            'size': position_size,
            'stop_loss': signal.get('stop_loss'),
            'take_profit': signal.get('take_profit'),
            'timestamp': get_timestamp(),
            'strategy': self.strategy_type,
            'confidence': signal.get('confidence', 0.5)
        }
        
        self.trade_history.append(trade)
        self.performance_metrics['total_trades'] += 1
        
        return trade

class MomentumStrategy(TradingStrategy):
    """Estratégia de Momentum Trading"""
    
    def __init__(self, risk_profile='moderate'):
        super().__init__('momentum', risk_profile)
        
    def generate_signal(self, market_analysis):
        """Gera sinal baseado em momentum"""
        prediction = market_analysis['prediction']
        signal_strength = market_analysis['signal']
        trend = market_analysis['trend']
        
        current_price = prediction['current_price']
        future_price = prediction['predictions'][-1]
        price_change = signal_strength['price_change_pct']
        
        # Lógica de momentum
        if (signal_strength['direction'] == 'Bullish' and 
            signal_strength['strength'] in ['Medium', 'Strong'] and
            trend['trend'] == 'Uptrend'):
            
            action = 'BUY'
            entry_price = current_price
            stop_loss = current_price * (1 - self.risk_settings['stop_loss'])
            take_profit = current_price * (1 + self.risk_settings['take_profit'])
            
        elif (signal_strength['direction'] == 'Bearish' and 
              signal_strength['strength'] in ['Medium', 'Strong'] and
              trend['trend'] == 'Downtrend'):
            
            action = 'SELL'
            entry_price = current_price
            stop_loss = current_price * (1 + self.risk_settings['stop_loss'])
            take_profit = current_price * (1 - self.risk_settings['take_profit'])
            
        else:
            action = 'HOLD'
            entry_price = current_price
            stop_loss = None
            take_profit = None
        
        return {
            'symbol': market_analysis['symbol'],
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': signal_strength['confidence'] / 100,
            'reasoning': f"Momentum {signal_strength['direction']} - {signal_strength['strength']}",
            'expected_return': price_change
        }

class MeanReversionStrategy(TradingStrategy):
    """Estratégia de Mean Reversion"""
    
    def __init__(self, risk_profile='moderate'):
        super().__init__('mean_reversion', risk_profile)
        
    def generate_signal(self, market_analysis):
        """Gera sinal baseado em reversão à média"""
        prediction = market_analysis['prediction']
        trend = market_analysis['trend']
        
        current_price = prediction['current_price']
        short_ma = trend['short_ma']
        long_ma = trend['long_ma']
        volatility = trend['volatility_pct']
        
        # Calcula desvio da média
        deviation = (current_price - long_ma) / long_ma
        
        # Lógica de mean reversion
        if deviation < -0.05 and volatility > 2.0:  # Preço muito abaixo da média
            action = 'BUY'
            entry_price = current_price
            stop_loss = current_price * (1 - self.risk_settings['stop_loss'])
            take_profit = long_ma  # Target é a média de longo prazo
            
        elif deviation > 0.05 and volatility > 2.0:  # Preço muito acima da média
            action = 'SELL'
            entry_price = current_price
            stop_loss = current_price * (1 + self.risk_settings['stop_loss'])
            take_profit = long_ma
            
        else:
            action = 'HOLD'
            entry_price = current_price
            stop_loss = None
            take_profit = None
        
        confidence = min(abs(deviation) * 10, 1.0)  # Maior desvio = maior confiança
        
        return {
            'symbol': market_analysis['symbol'],
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'reasoning': f"Mean Reversion - Desvio: {format_percentage(deviation * 100)}",
            'expected_return': -deviation * 50  # Expectativa de reversão
        }

class SwingStrategy(TradingStrategy):
    """Estratégia de Swing Trading"""
    
    def __init__(self, risk_profile='moderate'):
        super().__init__('swing', risk_profile)
        
    def generate_signal(self, market_analysis):
        """Gera sinal baseado em swing trading"""
        prediction = market_analysis['prediction']
        signal_strength = market_analysis['signal']
        trend = market_analysis['trend']
        
        current_price = prediction['current_price']
        price_change_24h = trend['price_change_24h']
        volatility = trend['volatility_pct']
        
        # Lógica de swing trading (busca reversões de curto prazo)
        if (price_change_24h < -3.0 and  # Queda significativa nas últimas 24h
            signal_strength['direction'] == 'Bullish' and  # Mas previsão é alta
            volatility > 1.5):  # Com volatilidade adequada
            
            action = 'BUY'
            entry_price = current_price
            stop_loss = current_price * (1 - self.risk_settings['stop_loss'])
            take_profit = current_price * (1 + self.risk_settings['take_profit'])
            
        elif (price_change_24h > 3.0 and  # Alta significativa nas últimas 24h
              signal_strength['direction'] == 'Bearish' and  # Mas previsão é baixa
              volatility > 1.5):
            
            action = 'SELL'
            entry_price = current_price
            stop_loss = current_price * (1 + self.risk_settings['stop_loss'])
            take_profit = current_price * (1 - self.risk_settings['take_profit'])
            
        else:
            action = 'HOLD'
            entry_price = current_price
            stop_loss = None
            take_profit = None
        
        # Confiança baseada na divergência entre movimento recente e previsão
        confidence = min(abs(price_change_24h) / 10, 1.0)
        
        return {
            'symbol': market_analysis['symbol'],
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'reasoning': f"Swing - Divergência detectada",
            'expected_return': signal_strength['price_change_pct']
        }

class MirrorStrategy(TradingStrategy):
    """Estratégia de Mirror Trading (copia instituições)"""
    
    def __init__(self, risk_profile='moderate'):
        super().__init__('mirror', risk_profile)
        self.institutional_confidence = {
            'berkshire': 0.85,
            'bridgewater': 0.80,
            'renaissance': 0.90
        }
        
    def simulate_institutional_signal(self, symbol):
        """Simula sinais de instituições famosas"""
        # Simula análise institucional
        institutional_signals = {}
        
        for institution, confidence in self.institutional_confidence.items():
            # Gera sinal baseado no "estilo" da instituição
            if institution == 'berkshire':  # Value investing
                signal = random.choice(['BUY', 'HOLD', 'HOLD', 'HOLD'])  # Conservador
            elif institution == 'bridgewater':  # Macro hedge fund
                signal = random.choice(['BUY', 'SELL', 'HOLD'])  # Balanceado
            elif institution == 'renaissance':  # Quant fund
                signal = random.choice(['BUY', 'SELL'])  # Mais ativo
                
            institutional_signals[institution] = {
                'signal': signal,
                'confidence': confidence * random.uniform(0.8, 1.0)
            }
        
        return institutional_signals
    
    def generate_signal(self, market_analysis):
        """Gera sinal baseado em mirror trading"""
        symbol = market_analysis['symbol']
        prediction = market_analysis['prediction']
        current_price = prediction['current_price']
        
        # Obtém sinais institucionais
        institutional_signals = self.simulate_institutional_signal(symbol)
        
        # Calcula consenso ponderado
        buy_weight = 0
        sell_weight = 0
        total_confidence = 0
        
        for institution, data in institutional_signals.items():
            confidence = data['confidence']
            signal = data['signal']
            
            if signal == 'BUY':
                buy_weight += confidence
            elif signal == 'SELL':
                sell_weight += confidence
                
            total_confidence += confidence
        
        # Determina ação baseada no consenso
        if buy_weight > sell_weight and buy_weight > 0.6:
            action = 'BUY'
            entry_price = current_price
            stop_loss = current_price * (1 - self.risk_settings['stop_loss'])
            take_profit = current_price * (1 + self.risk_settings['take_profit'])
            
        elif sell_weight > buy_weight and sell_weight > 0.6:
            action = 'SELL'
            entry_price = current_price
            stop_loss = current_price * (1 + self.risk_settings['stop_loss'])
            take_profit = current_price * (1 - self.risk_settings['take_profit'])
            
        else:
            action = 'HOLD'
            entry_price = current_price
            stop_loss = None
            take_profit = None
        
        confidence = total_confidence / len(institutional_signals)
        
        return {
            'symbol': symbol,
            'action': action,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'confidence': confidence,
            'reasoning': f"Mirror Trading - Consenso institucional",
            'institutional_signals': institutional_signals,
            'expected_return': random.uniform(-5, 15)  # Mirror trading pode ter retornos altos
        }

class StrategyManager:
    """Gerenciador de estratégias de trading"""
    
    def __init__(self):
        self.strategies = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'swing': SwingStrategy(),
            'mirror': MirrorStrategy()
        }
        self.active_strategy = 'momentum'
        
    def get_strategy(self, strategy_type):
        """Retorna estratégia específica"""
        return self.strategies.get(strategy_type)
    
    def set_active_strategy(self, strategy_type):
        """Define estratégia ativa"""
        if strategy_type in self.strategies:
            self.active_strategy = strategy_type
            return True
        return False
    
    def run_strategy_comparison(self, symbol, capital=100000):
        """Compara performance de todas as estratégias"""
        print(f"🏆 Comparando estratégias para {symbol}...")
        
        results = {}
        
        for strategy_name, strategy in self.strategies.items():
            print(f"📊 Testando {strategy_name}...")
            
            # Análise de mercado
            market_analysis = strategy.analyze_market(symbol)
            
            # Gera sinal
            signal = strategy.generate_signal(market_analysis)
            
            # Simula execução
            trade = strategy.execute_trade(signal, capital)
            
            results[strategy_name] = {
                'signal': signal,
                'trade': trade,
                'strategy_info': strategy.strategy_info,
                'expected_sharpe': strategy.strategy_info['sharpe_target']
            }
        
        # Encontra melhor estratégia
        best_strategy = max(results.keys(), 
                          key=lambda x: results[x]['expected_sharpe'])
        
        return {
            'symbol': symbol,
            'results': results,
            'best_strategy': best_strategy,
            'timestamp': get_timestamp()
        }
