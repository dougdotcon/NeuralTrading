#!/usr/bin/env python3
"""
🔥 AI FORECASTER WITH DEEPSEEK 🔥
Integração com Deepseek via OpenRouter para previsões reais de IA
Especializado em trading quantitativo e análise de séries temporais
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from openai import OpenAI
import numpy as np
import pandas as pd

# Adicionar pasta config ao path
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')
if config_path not in sys.path:
    sys.path.insert(0, config_path)

try:
    from api_config import get_api_config
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ api_config não disponível, usando variáveis de ambiente")

from .neural_config import get_timestamp, format_currency, format_percentage
from .real_data_collector import RealDataCollector


class AIForecaster:
    """
    Forecastador usando Deepseek via OpenRouter API
    Especializado em análise de séries temporais financeiras
    """
    
    def __init__(self, api_key: Optional[str] = None, use_real_data: bool = True):
        """
        Inicializa o forecastador de IA
        
        Args:
            api_key: Chave da API OpenRouter (ou None para usar configuração)
            use_real_data: Se True, usa dados reais de mercado
        """
        # Carregar configuração centralizada
        if CONFIG_AVAILABLE:
            self.config = get_api_config()
            self.api_key = api_key or self.config.get_api_key('openrouter')
            self.base_url = self.config.get('openrouter', 'base_url', 'https://openrouter.ai/api/v1')
            self.model = self.config.get('openrouter', 'model', 'deepseek/deepseek-r1-0528:free')
            self.timeout = self.config.get('openrouter', 'timeout', 30)
        else:
            # Fallback para variáveis de ambiente
            self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
            self.base_url = "https://openrouter.ai/api/v1"
            self.model = "deepseek/deepseek-r1-0528:free"
            self.timeout = 30
        
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
                print("✅ Cliente Deepseek/OpenRouter inicializado")
            except Exception as e:
                print(f"⚠️ Erro ao inicializar cliente OpenRouter: {e}")
                print("⚠️ Modo fallback ativado (simulação)")
        
        self.use_real_data = use_real_data
        self.data_collector = RealDataCollector() if use_real_data else None
        
        # Cache para evitar muitas requisições
        self.prediction_cache = {}
        self.cache_timeout = 300  # 5 minutos
        
        # Histórico de previsões
        self.prediction_history = []
    
    def is_available(self) -> bool:
        """Verifica se a API está disponível"""
        return self.client is not None and self.api_key is not None
    
    def prepare_market_context(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Prepara contexto completo de mercado para análise de IA
        Inclui dados históricos, indicadores técnicos e estatísticas
        """
        context = {
            'symbol': symbol,
            'timestamp': get_timestamp(),
            'data_source': 'real' if self.use_real_data else 'simulated'
        }
        
        # Obter dados históricos
        if self.use_real_data and self.data_collector:
            try:
                raw_data = self.data_collector.get_market_data(symbol, days=days)
                if raw_data is not None and len(raw_data) > 0:
                    # Dados básicos
                    prices = raw_data['close'].values if isinstance(raw_data, pd.DataFrame) else raw_data
                    volumes = raw_data['volume'].values if isinstance(raw_data, pd.DataFrame) and 'volume' in raw_data.columns else None
                    
                    context['data'] = {
                        'prices': prices.tolist()[-100:],  # Últimos 100 pontos
                        'volumes': volumes.tolist()[-100:] if volumes is not None else None,
                        'current_price': float(prices[-1]),
                        'min_price': float(np.min(prices)),
                        'max_price': float(np.max(prices)),
                        'avg_price': float(np.mean(prices)),
                        'volatility': float(np.std(prices) / np.mean(prices)),
                        'price_change_24h': float((prices[-1] - prices[-24]) / prices[-24] * 100) if len(prices) >= 24 else 0.0
                    }
                    
                    # Calcular indicadores técnicos
                    if isinstance(raw_data, pd.DataFrame):
                        indicators = self.data_collector.calculate_technical_indicators(raw_data)
                        context['technical_indicators'] = indicators
                    
                    return context
            except Exception as e:
                print(f"⚠️ Erro ao obter dados reais: {e}")
        
        # Fallback para dados simulados
        if self.use_real_data:
            print("⚠️ Usando dados simulados como fallback")
        
        # Gerar dados simulados básicos
        base_price = 100.0
        prices = base_price + np.cumsum(np.random.randn(100) * 0.02)
        
        context['data'] = {
            'prices': prices.tolist(),
            'current_price': float(prices[-1]),
            'min_price': float(np.min(prices)),
            'max_price': float(np.max(prices)),
            'avg_price': float(np.mean(prices)),
            'volatility': float(np.std(prices) / np.mean(prices)),
            'price_change_24h': float((prices[-1] - prices[-24]) / prices[-24] * 100) if len(prices) >= 24 else 0.0
        }
        
        return context
    
    def create_forecast_prompt(self, symbol: str, context: Dict[str, Any], horizon: int = 24) -> str:
        """
        Cria prompt especializado para análise de trading quantitativo
        Baseado em princípios de séries temporais e ML teórico
        """
        
        current_price = context['data']['current_price']
        volatility = context['data']['volatility']
        price_change_24h = context['data']['price_change_24h']
        
        # Preparar dados históricos formatados
        prices = context['data']['prices']
        recent_prices = prices[-20:]  # Últimas 20 observações
        
        # Indicadores técnicos se disponíveis
        indicators_info = ""
        if 'technical_indicators' in context:
            ti = context['technical_indicators']
            indicators_info = f"""
### Indicadores Técnicos:
- RSI (14): {ti.get('rsi', 'N/A'):.2f} - {'Sobrecomprado' if ti.get('rsi', 50) > 70 else 'Sobrevalorizado' if ti.get('rsi', 50) < 30 else 'Neutro'}
- SMA (20): ${ti.get('sma_20', current_price):.2f}
- SMA (50): ${ti.get('sma_50', current_price):.2f if 'sma_50' in ti else 'N/A'}
- Bollinger Bands: Superior ${ti.get('bb_upper', current_price * 1.05):.2f}, Inferior ${ti.get('bb_lower', current_price * 0.95):.2f}
"""
        
        prompt = f"""Você é um especialista em trading quantitativo e análise de séries temporais financeiras. Sua tarefa é analisar dados históricos de mercado e fornecer previsões fundamentadas.

## Contexto do Ativo: {symbol}

### Dados Atuais:
- Preço Atual: ${current_price:.2f}
- Volatilidade: {volatility*100:.2f}%
- Mudança 24h: {price_change_24h:.2f}%
- Preço Mínimo (período): ${context['data']['min_price']:.2f}
- Preço Máximo (período): ${context['data']['max_price']:.2f}
- Preço Médio (período): ${context['data']['avg_price']:.2f}

{indicators_info}

### Últimos 20 Preços (mais recentes):
{', '.join([f'${p:.2f}' for p in recent_prices])}

## Tarefa:

Analise os padrões históricos, tendências, e indicadores técnicos. Considere:

1. **Análise de Tendência**: Identifique tendência de alta, baixa ou lateral
2. **Padrões de Volatilidade**: Avalie períodos de alta/baixa volatilidade
3. **Indicadores Técnicos**: Use RSI, médias móveis, Bollinger Bands para contexto
4. **Análise Quantitativa**: Identifique possíveis níveis de suporte/resistência

Forneça previsões para os próximos {horizon} períodos (horas), incluindo:
- Previsão de preço esperado para cada período
- Intervalo de confiança (mínimo e máximo provável)
- Direção da tendência (alta/baixa/lateral)
- Força do sinal (Forte/Médio/Fraco)
- Raciocínio por trás da previsão

## Formato de Resposta Esperado (JSON):

{{
    "predictions": [
        {{"period": 1, "price": 100.50, "confidence_lower": 99.00, "confidence_upper": 102.00}},
        {{"period": 2, "price": 101.20, "confidence_lower": 99.50, "confidence_upper": 103.00}},
        ...
    ],
    "trend": "alta|baixa|lateral",
    "signal_strength": "Forte|Médio|Fraco",
    "reasoning": "Explicação detalhada da análise",
    "key_levels": {{
        "support": 95.00,
        "resistance": 105.00
    }},
    "risk_assessment": "baixo|médio|alto"
}}

Sua análise deve ser fundamentada em princípios de:
- Análise técnica quantitativa
- Séries temporais e forecasting
- Gerenciamento de risco
- Padrões estatísticos de mercado

Forneça uma resposta JSON válida seguindo exatamente o formato acima.
"""
        
        return prompt
    
    def forecast_with_ai(self, symbol: str, horizon: int = 24, days: int = 30) -> Dict[str, Any]:
        """
        Realiza previsão usando Deepseek via OpenRouter
        
        Args:
            symbol: Símbolo do ativo (ex: AAPL, BTC)
            horizon: Horizonte de previsão em períodos
            days: Quantidade de dias históricos para análise
            
        Returns:
            Dicionário com previsões e análise
        """
        if not self.is_available():
            raise ValueError("API Deepseek/OpenRouter não está disponível. Configure OPENROUTER_API_KEY")
        
        print(f"🤖 Gerando previsão com IA Deepseek para {symbol}...")
        print(f"📊 Horizonte: {horizon} períodos")
        
        # Verificar cache
        cache_key = f"{symbol}_{horizon}_{days}"
        if cache_key in self.prediction_cache:
            cached_result, cached_time = self.prediction_cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                print("📦 Usando previsão do cache")
                return cached_result
        
        # Preparar contexto de mercado
        context = self.prepare_market_context(symbol, days)
        
        # Criar prompt especializado
        prompt = self.create_forecast_prompt(symbol, context, horizon)
        
        try:
            start_time = time.time()
            
            # Chamar API Deepseek
            completion = self.client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://github.com/neural-trading",
                    "X-Title": "Neural Trading AI"
                },
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um especialista em trading quantitativo com profundo conhecimento em análise de séries temporais, machine learning e finanças. Forneça análises precisas e fundamentadas."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Baixa temperatura para previsões mais consistentes
                max_tokens=2000
            )
            
            inference_time = time.time() - start_time
            
            # Extrair resposta
            response_text = completion.choices[0].message.content
            
            # Tentar parsear JSON da resposta
            try:
                # Tentar extrair JSON da resposta (pode ter texto antes/depois)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                    ai_result = json.loads(json_text)
                else:
                    raise ValueError("JSON não encontrado na resposta")
            except json.JSONDecodeError:
                # Se falhar, tentar extrair informações manualmente
                print("⚠️ Erro ao parsear JSON, usando fallback")
                ai_result = self._parse_text_response(response_text, context, horizon)
            
            # Formatar resultado
            predictions = []
            for pred in ai_result.get('predictions', []):
                predictions.append(pred.get('price', context['data']['current_price']))
            
            result = {
                'symbol': symbol,
                'model': 'Deepseek-R1',
                'horizon': horizon,
                'current_price': context['data']['current_price'],
                'predictions': predictions[:horizon],  # Garantir tamanho correto
                'confidence_intervals': [
                    (
                        pred.get('confidence_lower', pred.get('price', context['data']['current_price']) * 0.95),
                        pred.get('confidence_upper', pred.get('price', context['data']['current_price']) * 1.05)
                    ) for pred in ai_result.get('predictions', [])[:horizon]
                ],
                'trend': ai_result.get('trend', 'lateral'),
                'signal_strength': ai_result.get('signal_strength', 'Médio'),
                'reasoning': ai_result.get('reasoning', 'Análise não disponível'),
                'key_levels': ai_result.get('key_levels', {}),
                'risk_assessment': ai_result.get('risk_assessment', 'médio'),
                'metrics': {
                    'inference_time_ms': inference_time * 1000,
                    'api_calls': 1
                },
                'timestamp': get_timestamp(),
                'data_source': context['data_source']
            }
            
            # Cache resultado
            self.prediction_cache[cache_key] = (result, time.time())
            self.prediction_history.append(result)
            
            print(f"✅ Previsão gerada em {inference_time*1000:.2f}ms")
            
            return result
            
        except Exception as e:
            print(f"❌ Erro ao gerar previsão com IA: {e}")
            raise
    
    def _parse_text_response(self, text: str, context: Dict[str, Any], horizon: int) -> Dict[str, Any]:
        """
        Fallback: Extrai informações de resposta em texto livre
        """
        current_price = context['data']['current_price']
        
        # Gerar previsões básicas baseadas em tendência
        trend_factor = 1.001  # Pequena tendência positiva
        predictions = []
        
        for i in range(horizon):
            price = current_price * (trend_factor ** i)
            predictions.append({
                'period': i + 1,
                'price': price,
                'confidence_lower': price * 0.97,
                'confidence_upper': price * 1.03
            })
        
        return {
            'predictions': predictions,
            'trend': 'lateral',
            'signal_strength': 'Médio',
            'reasoning': text[:500],  # Primeiros 500 caracteres
            'key_levels': {
                'support': current_price * 0.95,
                'resistance': current_price * 1.05
            },
            'risk_assessment': 'médio'
        }
    
    def get_signal_strength(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula força do sinal de trading baseado na previsão de IA
        """
        predictions = prediction_result['predictions']
        current_price = prediction_result['current_price']
        
        if not predictions:
            return {
                'direction': 'Neutral',
                'strength': 'Weak',
                'price_change_pct': 0.0,
                'confidence': 50.0
            }
        
        # Preço futuro esperado
        future_price = predictions[-1] if predictions else current_price
        price_change = (future_price - current_price) / current_price
        
        # Determinar direção
        if price_change > 0.03:  # > 3%
            direction = 'Bullish'
        elif price_change < -0.03:  # < -3%
            direction = 'Bearish'
        else:
            direction = 'Neutral'
        
        # Determinar força (usar signal_strength da IA se disponível)
        ai_strength = prediction_result.get('signal_strength', 'Médio')
        strength_map = {
            'Forte': 'Strong',
            'Strong': 'Strong',
            'Médio': 'Medium',
            'Medium': 'Medium',
            'Fraco': 'Weak',
            'Weak': 'Weak'
        }
        strength = strength_map.get(ai_strength, 'Medium')
        
        # Confiança baseada em múltiplos fatores
        confidence = 70.0  # Base
        if abs(price_change) > 0.05:
            confidence += 10
        if prediction_result.get('risk_assessment') == 'baixo':
            confidence += 5
        
        return {
            'direction': direction,
            'strength': strength,
            'price_change_pct': price_change * 100,
            'confidence': min(confidence, 95.0)
        }

