# Integração com Deepseek IA via OpenRouter

## Visão Geral

O sistema agora suporta **previsões reais de IA** usando Deepseek através da API OpenRouter. Esta integração permite análises quantitativas avançadas de séries temporais financeiras com base em aprendizado de máquina teórico.

## Como Funciona

### 1. Coleta de Dados
- O sistema coleta dados reais de mercado (Yahoo Finance, CoinGecko, etc.)
- Calcula indicadores técnicos (RSI, SMA, Bollinger Bands)
- Prepara contexto estatístico completo

### 2. Preparação do Prompt
- Cria prompt especializado para trading quantitativo
- Inclui dados históricos, indicadores técnicos, volatilidade
- Baseado em princípios de séries temporais e ML teórico (documentação FT-MLA-001)

### 3. Análise com Deepseek
- Envia contexto para Deepseek via OpenRouter API
- Recebe análise quantitativa detalhada
- Inclui previsões, tendências, níveis de suporte/resistência

### 4. Integração no Sistema
- Converte resultado da IA para formato padrão do sistema
- Mantém compatibilidade com código existente
- Fallback automático se API não disponível

## Configuração

### 1. Obter API Key do OpenRouter

1. Acesse: https://openrouter.ai/keys
2. Crie uma conta (gratuita)
3. Gere uma API key

### 2. Configurar Variável de Ambiente

#### Windows (PowerShell):
```powershell
$env:OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

#### Windows (CMD):
```cmd
set OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

#### Linux/Mac:
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-key-here"
```

#### Usando arquivo .env (Recomendado):
1. Copie `.env.example` para `.env` na raiz do projeto
2. Edite `.env`:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 3. Instalar Dependências

```bash
pip install -r requirements.txt
```

Isso instalará:
- `openai>=1.0.0` - Cliente para OpenRouter
- `python-dotenv>=1.0.0` - Carregamento de .env

## Uso

### Ativação Automática

O sistema detecta automaticamente se `OPENROUTER_API_KEY` está configurada:
- ✅ **Se configurada**: Usa Deepseek para previsões reais
- ⚠️ **Se não configurada**: Usa modo padrão (simulação)

### Como Usar

1. Configure a API key (veja acima)
2. Execute o sistema normalmente:
   ```bash
   python start.py
   ```
3. Vá para "Previsão Neural"
4. O sistema automaticamente usará Deepseek se disponível

### Exemplo de Previsão com IA

```
🤖 Usando Deepseek IA para previsão de AAPL...
📊 Horizonte: 24 períodos
🌐 Obtendo dados reais para AAPL...
📈 Coletando dados de AAPL via Yahoo Finance...
✅ Dados reais obtidos: 152 pontos
📊 Indicadores técnicos calculados para AAPL
🤖 Gerando previsão com IA Deepseek para AAPL...
✅ Previsão gerada em 1234.56ms

📊 RESULTADO DA PREVISÃO IA:
💰 PREÇO ATUAL: $203.94 (REAL)
🔮 PREVISÃO FINAL: $206.15 (+1.08%)

📈 INSIGHTS DA IA:
🔄 Tendência: Alta
💪 Força do Sinal: Forte
🎯 Suporte: $200.00
🎯 Resistência: $210.00
⚠️ Risco: Médio

💭 Raciocínio da IA:
Análise dos últimos 100 pontos mostra tendência de alta consolidada...
[explicação detalhada]
```

## Modelos Deepseek Disponíveis

### Gratuito (Recomendado para testes)
- **deepseek/deepseek-r1-0528:free**: Modelo gratuito com boa qualidade

### Pago (Melhor performance)
- **deepseek/deepseek-chat**: Mais rápido, menor latência
- **deepseek/deepseek-r1**: Melhor qualidade, análise mais profunda

Para alterar o modelo, edite `ai_forecaster.py` e mude a variável `self.model`.

## Estrutura do Prompt

O prompt enviado ao Deepseek inclui:

1. **Dados Históricos**: Últimos 100 pontos de preço
2. **Estatísticas**: Volatilidade, min/max, média
3. **Indicadores Técnicos**: 
   - RSI (14)
   - SMA (20, 50)
   - Bollinger Bands
4. **Contexto de Mercado**: Mudança 24h, tendências
5. **Instruções Especializadas**: Baseadas em ML teórico

## Resposta Esperada

O Deepseek retorna análise em JSON com:

```json
{
    "predictions": [
        {"period": 1, "price": 100.50, "confidence_lower": 99.00, "confidence_upper": 102.00},
        ...
    ],
    "trend": "alta|baixa|lateral",
    "signal_strength": "Forte|Médio|Fraco",
    "reasoning": "Explicação detalhada",
    "key_levels": {
        "support": 95.00,
        "resistance": 105.00
    },
    "risk_assessment": "baixo|médio|alto"
}
```

## Fallback Automático

Se a API falhar ou não estiver disponível:
- ✅ Sistema continua funcionando normalmente
- ✅ Usa modo padrão (simulação)
- ✅ Não interrompe operações
- ⚠️ Mostra mensagem informativa

## Vantagens da Integração

### 1. Análise Real de IA
- ✅ Previsões baseadas em aprendizado de máquina real
- ✅ Considera padrões complexos de séries temporais
- ✅ Análise quantitativa fundamentada

### 2. Contexto Rico
- ✅ Usa dados reais de mercado
- ✅ Inclui indicadores técnicos calculados
- ✅ Considera volatilidade e tendências

### 3. Integração Seamless
- ✅ Compatível com código existente
- ✅ Fallback automático
- ✅ Sem breaking changes

### 4. Insights Avançados
- ✅ Raciocínio explicável
- ✅ Níveis de suporte/resistência
- ✅ Avaliação de risco
- ✅ Força de sinal

## Limitações

1. **Rate Limits**: OpenRouter tem limites de requisições (varia por plano)
2. **Latência**: Requisições de API adicionam ~1-3 segundos
3. **Custos**: Modelos pagos podem ter custos (gratuito disponível)
4. **Dependência de Internet**: Requer conexão ativa

## Cache Inteligente

O sistema implementa cache para otimizar:
- Previsões são cacheadas por 5 minutos
- Evita requisições desnecessárias
- Reduz latência e custos

## Troubleshooting

### API não funciona

1. Verifique se `OPENROUTER_API_KEY` está configurada
2. Teste a chave diretamente:
   ```bash
   python -c "import os; print(os.getenv('OPENROUTER_API_KEY'))"
   ```
3. Verifique conexão com internet
4. Verifique logs de erro no console

### Previsões não aparecem

1. Verifique se modo IA está ativado (aparece no menu)
2. Verifique logs para erros
3. Tente novamente (pode ser rate limit)

### Erro de parsing JSON

- O sistema tem fallback automático
- Tenta extrair informações mesmo se JSON mal formatado
- Usa análise básica como backup

## Próximos Passos

- [ ] Suporte a múltiplos modelos (escolha no menu)
- [ ] Cache persistente em arquivo
- [ ] Análise comparativa IA vs simulação
- [ ] Métricas de acurácia da IA
- [ ] Batch predictions otimizado

## Referências

- **OpenRouter**: https://openrouter.ai
- **Deepseek**: https://www.deepseek.com
- **Documentação ML Teórico**: `docs/FT-MLA-001-pt-aprendizado-maquina-teorico-v1.0.md`

