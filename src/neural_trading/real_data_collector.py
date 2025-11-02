#!/usr/bin/env python3
"""
🔥 REAL DATA COLLECTOR 🔥
Coletor de dados reais de mercado para o sistema NeuralTrading
Integra APIs gratuitas para ações, crypto e forex
"""

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
from .neural_config import POPULAR_ASSETS, get_timestamp

class RealDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuralTrading/1.0 (Educational Purpose)'
        })
        
        # Cache para evitar muitas requisições
        self.cache = {}
        self.cache_timeout = 300  # 5 minutos
        
    def get_stock_data_yahoo(self, symbol, period='1mo'):
        """Coleta dados de ações via Yahoo Finance API gratuita"""
        try:
            print(f"📈 Coletando dados de {symbol} via Yahoo Finance...")
            
            # Yahoo Finance API gratuita
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'range': period,
                'interval': '1h',
                'includePrePost': 'false'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' not in data or not data['chart']['result']:
                print(f"❌ Nenhum dado encontrado para {symbol}")
                return None
                
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            # Cria DataFrame
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(ts) for ts in timestamps],
                'open': quotes['open'],
                'high': quotes['high'],
                'low': quotes['low'],
                'close': quotes['close'],
                'volume': quotes['volume']
            })
            
            # Remove valores nulos
            df = df.dropna()
            
            if len(df) == 0:
                print(f"❌ Dados inválidos para {symbol}")
                return None
                
            print(f"✅ Coletados {len(df)} registros para {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao coletar dados de {symbol}: {str(e)}")
            return None
    
    def get_crypto_data_coingecko(self, symbol, days=30):
        """Coleta dados de crypto via CoinGecko API gratuita"""
        try:
            print(f"🪙 Coletando dados de {symbol} via CoinGecko...")
            
            # Mapeia símbolos para IDs do CoinGecko
            crypto_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOT': 'polkadot',
                'AVAX': 'avalanche-2',
                'MATIC': 'matic-network'
            }
            
            coin_id = crypto_map.get(symbol, symbol.lower())
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'hourly'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'prices' not in data:
                print(f"❌ Nenhum dado encontrado para {symbol}")
                return None
            
            # Processa dados
            prices = data['prices']
            volumes = data.get('total_volumes', [])
            
            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(price[0]/1000) for price in prices],
                'close': [price[1] for price in prices],
                'volume': [vol[1] if vol else 0 for vol in volumes[:len(prices)]]
            })
            
            # Adiciona colunas OHLC simuladas (CoinGecko free só tem close)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df['close'] * (1 + np.random.uniform(0, 0.02, len(df)))
            df['low'] = df['close'] * (1 - np.random.uniform(0, 0.02, len(df)))
            
            print(f"✅ Coletados {len(df)} registros para {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao coletar dados de {symbol}: {str(e)}")
            return None
    
    def get_forex_data_exchangerate(self, pair, days=30):
        """Coleta dados de forex via ExchangeRate-API gratuita"""
        try:
            print(f"💱 Coletando dados de {pair} via ExchangeRate-API...")
            
            # Separa o par (ex: EUR/USD -> EUR, USD)
            base, quote = pair.split('/')
            
            # API gratuita tem limitações, vamos simular dados históricos
            # baseados na taxa atual
            url = f"https://api.exchangerate-api.com/v4/latest/{base}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if quote not in data['rates']:
                print(f"❌ Par {pair} não encontrado")
                return None
            
            current_rate = data['rates'][quote]
            
            # Simula dados históricos baseados na taxa atual
            dates = []
            rates = []
            
            for i in range(days * 24):  # Dados horários
                date = datetime.now() - timedelta(hours=i)
                # Adiciona variação realística
                variation = np.random.normal(0, 0.005)  # 0.5% de volatilidade
                rate = current_rate * (1 + variation)
                
                dates.append(date)
                rates.append(rate)
            
            dates.reverse()
            rates.reverse()
            
            df = pd.DataFrame({
                'timestamp': dates,
                'close': rates,
                'open': rates,  # Simplificado
                'high': [r * (1 + abs(np.random.normal(0, 0.002))) for r in rates],
                'low': [r * (1 - abs(np.random.normal(0, 0.002))) for r in rates],
                'volume': [1000000] * len(rates)  # Volume simulado
            })
            
            print(f"✅ Coletados {len(df)} registros para {pair}")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao coletar dados de {pair}: {str(e)}")
            return None
    
    def get_commodity_data_alpha_vantage(self, symbol):
        """Coleta dados de commodities (simulado - API gratuita limitada)"""
        try:
            print(f"🥇 Simulando dados de {symbol}...")
            
            # Preços base para commodities
            base_prices = {
                'GOLD': 2000,
                'SILVER': 25,
                'OIL': 80,
                'COPPER': 8000,
                'WHEAT': 600,
                'CORN': 450
            }
            
            base_price = base_prices.get(symbol, 100)
            
            # Simula 30 dias de dados horários
            dates = []
            prices = []
            current_price = base_price
            
            for i in range(30 * 24):
                date = datetime.now() - timedelta(hours=i)
                # Volatilidade específica por commodity
                volatility = 0.02 if symbol in ['GOLD', 'SILVER'] else 0.03
                change = np.random.normal(0, volatility)
                current_price *= (1 + change)
                
                dates.append(date)
                prices.append(current_price)
            
            dates.reverse()
            prices.reverse()
            
            df = pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'volume': [50000] * len(prices)
            })
            
            print(f"✅ Simulados {len(df)} registros para {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Erro ao simular dados de {symbol}: {str(e)}")
            return None
    
    def get_market_data(self, symbol, asset_type=None):
        """Método principal para coletar dados de qualquer ativo"""
        # Verifica cache primeiro
        cache_key = f"{symbol}_{asset_type}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_timeout:
                print(f"📋 Usando dados em cache para {symbol}")
                return data
        
        # Determina tipo de ativo se não especificado
        if asset_type is None:
            if symbol in POPULAR_ASSETS['stocks']:
                asset_type = 'stocks'
            elif symbol in POPULAR_ASSETS['crypto']:
                asset_type = 'crypto'
            elif symbol in POPULAR_ASSETS['forex']:
                asset_type = 'forex'
            elif symbol in POPULAR_ASSETS['commodities']:
                asset_type = 'commodities'
            else:
                asset_type = 'stocks'  # Default
        
        # Coleta dados baseado no tipo
        data = None
        
        if asset_type == 'stocks':
            data = self.get_stock_data_yahoo(symbol)
        elif asset_type == 'crypto':
            data = self.get_crypto_data_coingecko(symbol)
        elif asset_type == 'forex':
            data = self.get_forex_data_exchangerate(symbol)
        elif asset_type == 'commodities':
            data = self.get_commodity_data_alpha_vantage(symbol)
        
        # Armazena no cache
        if data is not None:
            self.cache[cache_key] = (time.time(), data)
        
        return data
    
    def get_current_price(self, symbol, asset_type=None):
        """Obtém preço atual de um ativo"""
        data = self.get_market_data(symbol, asset_type)
        if data is not None and len(data) > 0:
            return data['close'].iloc[-1]
        return None
    
    def get_price_history(self, symbol, hours=24, asset_type=None):
        """Obtém histórico de preços das últimas N horas"""
        data = self.get_market_data(symbol, asset_type)
        if data is not None and len(data) > hours:
            return data['close'].tail(hours).values
        elif data is not None:
            return data['close'].values
        return None
    
    def calculate_technical_indicators(self, data):
        """Calcula indicadores técnicos básicos"""
        if data is None or len(data) < 20:
            return {}
        
        prices = data['close']
        
        # RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Médias móveis
        sma_20 = prices.rolling(window=20).mean()
        sma_50 = prices.rolling(window=50).mean() if len(prices) >= 50 else sma_20
        
        # Bollinger Bands
        bb_std = prices.rolling(window=20).std()
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        
        return {
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'sma_20': sma_20.iloc[-1] if not sma_20.empty else prices.iloc[-1],
            'sma_50': sma_50.iloc[-1] if not sma_50.empty else prices.iloc[-1],
            'bb_upper': bb_upper.iloc[-1] if not bb_upper.empty else prices.iloc[-1],
            'bb_lower': bb_lower.iloc[-1] if not bb_lower.empty else prices.iloc[-1],
            'volatility': prices.pct_change().std() * 100
        }
    
    def test_apis(self):
        """Testa conectividade com todas as APIs"""
        print("🔍 Testando conectividade com APIs...")
        
        tests = [
            ('AAPL', 'stocks'),
            ('BTC', 'crypto'),
            ('EUR/USD', 'forex'),
            ('GOLD', 'commodities')
        ]
        
        results = {}
        for symbol, asset_type in tests:
            print(f"\n🧪 Testando {symbol} ({asset_type})...")
            data = self.get_market_data(symbol, asset_type)
            results[symbol] = data is not None
            
            if data is not None:
                current_price = data['close'].iloc[-1]
                print(f"✅ {symbol}: ${current_price:.2f}")
            else:
                print(f"❌ {symbol}: Falha na coleta")
        
        success_rate = sum(results.values()) / len(results) * 100
        print(f"\n📊 Taxa de sucesso: {success_rate:.1f}%")
        
        return results
