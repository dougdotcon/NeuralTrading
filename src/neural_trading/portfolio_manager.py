#!/usr/bin/env python3
"""
🔥 PORTFOLIO MANAGER 🔥
Gerenciador de portfólio para o sistema NeuralTrading
Controla posições, risco e performance
"""

import random
import time
import numpy as np
from datetime import datetime, timedelta
from .neural_config import (
    DEFAULT_PORTFOLIO, RISK_SETTINGS, POPULAR_ASSETS,
    get_timestamp, format_currency, format_percentage
)

class Portfolio:
    """Classe para gerenciar um portfólio de trading"""
    
    def __init__(self, initial_capital=100000, risk_profile='moderate'):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_profile = risk_profile
        self.risk_settings = RISK_SETTINGS[risk_profile]
        
        # Posições ativas
        self.positions = {}
        
        # Histórico
        self.trade_history = []
        self.equity_curve = [initial_capital]
        self.daily_returns = []
        
        # Métricas de performance
        self.metrics = {
            'total_return': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Configurações
        self.created_at = get_timestamp()
        self.last_updated = get_timestamp()
        
    def add_position(self, symbol, action, price, size, strategy='manual'):
        """Adiciona nova posição ao portfólio"""
        position_value = price * size
        
        # Verifica se há capital suficiente
        if action == 'BUY' and position_value > self.current_capital:
            return False, "Capital insuficiente"
        
        # Verifica limites de risco
        position_risk = position_value / self.current_capital
        if position_risk > self.risk_settings['max_position_size']:
            return False, f"Posição excede limite de risco ({format_percentage(self.risk_settings['max_position_size'] * 100)})"
        
        # Cria posição
        position_id = f"{symbol}_{int(time.time())}"
        position = {
            'id': position_id,
            'symbol': symbol,
            'action': action,
            'entry_price': price,
            'size': size,
            'value': position_value,
            'strategy': strategy,
            'opened_at': get_timestamp(),
            'unrealized_pnl': 0.0,
            'unrealized_pnl_pct': 0.0
        }
        
        self.positions[position_id] = position
        
        # Atualiza capital
        if action == 'BUY':
            self.current_capital -= position_value
        
        self.last_updated = get_timestamp()
        return True, f"Posição {action} {symbol} adicionada com sucesso"
    
    def close_position(self, position_id, current_price):
        """Fecha uma posição"""
        if position_id not in self.positions:
            return False, "Posição não encontrada"
        
        position = self.positions[position_id]
        
        # Calcula P&L
        if position['action'] == 'BUY':
            pnl = (current_price - position['entry_price']) * position['size']
        else:  # SELL
            pnl = (position['entry_price'] - current_price) * position['size']
        
        pnl_pct = pnl / position['value'] * 100
        
        # Atualiza capital
        if position['action'] == 'BUY':
            self.current_capital += current_price * position['size']
        else:
            self.current_capital += position['value'] + pnl
        
        # Registra trade no histórico
        trade_record = {
            'position_id': position_id,
            'symbol': position['symbol'],
            'action': position['action'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'size': position['size'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'strategy': position['strategy'],
            'opened_at': position['opened_at'],
            'closed_at': get_timestamp(),
            'duration': self._calculate_duration(position['opened_at'])
        }
        
        self.trade_history.append(trade_record)
        
        # Remove posição
        del self.positions[position_id]
        
        # Atualiza métricas
        self._update_metrics()
        
        self.last_updated = get_timestamp()
        return True, f"Posição fechada - P&L: {format_currency(pnl)} ({format_percentage(pnl_pct)})"
    
    def update_positions(self, market_prices):
        """Atualiza P&L não realizado das posições"""
        for position_id, position in self.positions.items():
            symbol = position['symbol']
            if symbol in market_prices:
                current_price = market_prices[symbol]
                
                if position['action'] == 'BUY':
                    unrealized_pnl = (current_price - position['entry_price']) * position['size']
                else:  # SELL
                    unrealized_pnl = (position['entry_price'] - current_price) * position['size']
                
                position['unrealized_pnl'] = unrealized_pnl
                position['unrealized_pnl_pct'] = unrealized_pnl / position['value'] * 100
        
        self.last_updated = get_timestamp()
    
    def get_portfolio_value(self, market_prices):
        """Calcula valor total do portfólio"""
        total_value = self.current_capital
        
        for position in self.positions.values():
            symbol = position['symbol']
            if symbol in market_prices:
                current_price = market_prices[symbol]
                if position['action'] == 'BUY':
                    position_value = current_price * position['size']
                else:
                    position_value = position['value'] + position['unrealized_pnl']
                total_value += position_value
        
        return total_value
    
    def get_risk_metrics(self):
        """Calcula métricas de risco do portfólio"""
        if len(self.daily_returns) < 2:
            return {
                'volatility': 0.0,
                'var_95': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        returns = np.array(self.daily_returns)
        
        # Volatilidade anualizada
        volatility = np.std(returns) * np.sqrt(252) * 100
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5) * self.current_capital
        
        # Sharpe Ratio
        risk_free_rate = 0.02  # 2% ao ano
        excess_returns = returns - (risk_free_rate / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum Drawdown
        equity_curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def get_allocation(self):
        """Retorna alocação atual do portfólio"""
        total_value = sum(pos['value'] for pos in self.positions.values())
        
        if total_value == 0:
            return {'cash': 100.0}
        
        allocation = {}
        
        # Calcula alocação por ativo
        for position in self.positions.values():
            symbol = position['symbol']
            weight = (position['value'] / total_value) * 100
            
            if symbol in allocation:
                allocation[symbol] += weight
            else:
                allocation[symbol] = weight
        
        # Adiciona cash
        cash_weight = (self.current_capital / (total_value + self.current_capital)) * 100
        allocation['cash'] = cash_weight
        
        return allocation
    
    def rebalance_portfolio(self, target_allocation, market_prices):
        """Rebalanceia portfólio para alocação alvo"""
        current_value = self.get_portfolio_value(market_prices)
        rebalance_trades = []
        
        for symbol, target_weight in target_allocation.items():
            if symbol == 'cash':
                continue
                
            target_value = current_value * (target_weight / 100)
            current_value_symbol = sum(
                pos['value'] for pos in self.positions.values() 
                if pos['symbol'] == symbol
            )
            
            difference = target_value - current_value_symbol
            
            if abs(difference) > current_value * 0.01:  # Só rebalanceia se diferença > 1%
                if difference > 0:  # Precisa comprar
                    action = 'BUY'
                    size = difference / market_prices[symbol]
                else:  # Precisa vender
                    action = 'SELL'
                    size = abs(difference) / market_prices[symbol]
                
                rebalance_trades.append({
                    'symbol': symbol,
                    'action': action,
                    'size': size,
                    'price': market_prices[symbol],
                    'reason': 'rebalance'
                })
        
        return rebalance_trades
    
    def _update_metrics(self):
        """Atualiza métricas de performance"""
        if not self.trade_history:
            return
        
        # Calcula métricas básicas
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        losing_trades = total_trades - winning_trades
        
        total_pnl = sum(trade['pnl'] for trade in self.trade_history)
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit Factor
        gross_profit = sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] > 0)
        gross_loss = abs(sum(trade['pnl'] for trade in self.trade_history if trade['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        self.metrics.update({
            'total_return': total_pnl,
            'total_return_pct': total_return_pct,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        })
    
    def _calculate_duration(self, start_time):
        """Calcula duração entre timestamps"""
        try:
            start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
            end = datetime.strptime(get_timestamp(), '%Y-%m-%d %H:%M:%S')
            duration = end - start
            return str(duration)
        except:
            return "N/A"
    
    def get_summary(self):
        """Retorna resumo do portfólio"""
        risk_metrics = self.get_risk_metrics()
        allocation = self.get_allocation()
        
        return {
            'capital': {
                'initial': self.initial_capital,
                'current': self.current_capital,
                'total_return': self.metrics['total_return'],
                'total_return_pct': self.metrics['total_return_pct']
            },
            'positions': {
                'active_positions': len(self.positions),
                'total_trades': self.metrics['total_trades'],
                'win_rate': self.metrics['win_rate']
            },
            'risk': {
                'profile': self.risk_profile,
                'volatility': risk_metrics['volatility'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'max_drawdown': risk_metrics['max_drawdown'],
                'var_95': risk_metrics['var_95']
            },
            'allocation': allocation,
            'last_updated': self.last_updated
        }

class PortfolioManager:
    """Gerenciador de múltiplos portfólios"""
    
    def __init__(self):
        self.portfolios = {}
        self.active_portfolio = None
        
    def create_portfolio(self, name, initial_capital=100000, risk_profile='moderate'):
        """Cria novo portfólio"""
        portfolio = Portfolio(initial_capital, risk_profile)
        self.portfolios[name] = portfolio
        
        if self.active_portfolio is None:
            self.active_portfolio = name
            
        return portfolio
    
    def get_portfolio(self, name):
        """Retorna portfólio específico"""
        return self.portfolios.get(name)
    
    def set_active_portfolio(self, name):
        """Define portfólio ativo"""
        if name in self.portfolios:
            self.active_portfolio = name
            return True
        return False
    
    def get_active_portfolio(self):
        """Retorna portfólio ativo"""
        if self.active_portfolio:
            return self.portfolios[self.active_portfolio]
        return None
    
    def compare_portfolios(self):
        """Compara performance de todos os portfólios"""
        comparison = {}
        
        for name, portfolio in self.portfolios.items():
            summary = portfolio.get_summary()
            comparison[name] = {
                'return_pct': summary['capital']['total_return_pct'],
                'sharpe_ratio': summary['risk']['sharpe_ratio'],
                'max_drawdown': summary['risk']['max_drawdown'],
                'total_trades': summary['positions']['total_trades'],
                'win_rate': summary['positions']['win_rate']
            }
        
        return comparison
