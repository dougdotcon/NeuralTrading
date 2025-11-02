# Configuração de APIs

## Visão Geral

O sistema usa configuração centralizada na pasta `config/` para gerenciar todas as chaves de API e configurações de serviços.

## Estrutura de Configuração

```
config/
├── __init__.py              # Módulo de configuração
├── api_config.py            # Gerenciador de configurações
├── api_keys.json            # Suas chaves (não commitado)
├── api_keys.json.example    # Exemplo de configuração
└── README.md                # Documentação
```

## Como Configurar

### 1. Criar Arquivo de Configuração

```bash
# Copie o arquivo de exemplo
cp config/api_keys.json.example config/api_keys.json
```

### 2. Editar Configurações

Abra `config/api_keys.json` e configure suas chaves:

```json
{
  "openrouter": {
    "api_key": "sk-or-v1-your-openrouter-key",
    "model": "deepseek/deepseek-r1-0528:free",
    "timeout": 30
  },
  "coingecko": {
    "api_key": "your-coingecko-key-optional",
    "rate_limit": 50
  },
  "exchangerate": {
    "api_key": "optional"
  },
  "alpha_vantage": {
    "api_key": "your-alpha-vantage-key"
  },
  "system": {
    "use_ai": true,
    "use_real_data": true,
    "cache_enabled": true
  }
}
```

## APIs Suportadas

### 1. OpenRouter (Deepseek) - **Obrigatória para IA**

**Onde obter:** https://openrouter.ai/keys

**Configuração:**
```json
{
  "openrouter": {
    "api_key": "sk-or-v1-your-key",
    "model": "deepseek/deepseek-r1-0528:free"
  }
}
```

**Modelos disponíveis:**
- `deepseek/deepseek-r1-0528:free` - Gratuito (recomendado)
- `deepseek/deepseek-chat` - Pago, mais rápido
- `deepseek/deepseek-r1` - Pago, melhor qualidade

### 2. Yahoo Finance - **Gratuita, sem API key**

Não requer configuração. Funciona automaticamente.

### 3. CoinGecko - **Opcional**

**Onde obter:** https://www.coingecko.com/api/pricing

**Configuração:**
```json
{
  "coingecko": {
    "api_key": "your-key-optional",
    "rate_limit": 50
  }
}
```

**Nota:** Funciona sem API key, mas com rate limit mais baixo (10-50 req/min).

### 4. ExchangeRate-API - **Gratuita, sem API key**

Não requer configuração para uso básico.

### 5. Alpha Vantage - **Opcional (futuro)**

**Onde obter:** https://www.alphavantage.co/support/#api-key

**Configuração:**
```json
{
  "alpha_vantage": {
    "api_key": "your-key"
  }
}
```

## Uso Programático

### Exemplo: Acessar Configuração

```python
from config.api_config import get_api_config

config = get_api_config()

# Obter API key
openrouter_key = config.get_api_key('openrouter')

# Obter configuração específica
model = config.get('openrouter', 'model')
timeout = config.get('openrouter', 'timeout', 30)

# Verificar se serviço está disponível
if config.has_api_key('openrouter'):
    print("OpenRouter configurado!")
```

### Exemplo: Atualizar Configuração

```python
from config.api_config import get_api_config

config = get_api_config()
config.set('openrouter', 'model', 'deepseek/deepseek-chat')
```

## Variáveis de Ambiente (Fallback)

O sistema também suporta variáveis de ambiente como fallback:

- `OPENROUTER_API_KEY`
- `COINGECKO_API_KEY`
- `EXCHANGERATE_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

**Prioridade:**
1. Arquivo `config/api_keys.json` (maior prioridade)
2. Variáveis de ambiente
3. Valores padrão

## Segurança

⚠️ **IMPORTANTE:**

- ❌ **NUNCA** commite `config/api_keys.json` no Git
- ✅ O arquivo está no `.gitignore`
- ✅ Use apenas `api_keys.json.example` para documentação
- ✅ Mantenha suas chaves privadas

## Troubleshooting

### Configuração não carrega

1. Verifique se o arquivo existe: `config/api_keys.json`
2. Verifique formato JSON (valide em https://jsonlint.com)
3. Verifique permissões do arquivo

### API key não funciona

1. Verifique se a chave está correta no arquivo
2. Teste a chave diretamente na API
3. Verifique logs de erro do sistema

### Usa variável de ambiente mesmo com arquivo

1. Verifique se o arquivo está no caminho correto
2. Reinicie o sistema para recarregar configuração
3. Use `reload_config()` em código Python

## Recarregar Configuração

Se você atualizar `config/api_keys.json` durante execução:

```python
from config.api_config import reload_config

config = reload_config()  # Recarrega do arquivo
```

## Estrutura Completa do Arquivo

```json
{
  "openrouter": {
    "api_key": "string",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "deepseek/deepseek-r1-0528:free",
    "timeout": 30,
    "max_retries": 3
  },
  "yahoo_finance": {
    "base_url": "https://query1.finance.yahoo.com/v8/finance/chart",
    "timeout": 10,
    "retry_count": 3,
    "cache_timeout": 300
  },
  "coingecko": {
    "base_url": "https://api.coingecko.com/api/v3",
    "api_key": "",
    "timeout": 10,
    "retry_count": 3,
    "cache_timeout": 300,
    "rate_limit": 50
  },
  "exchangerate": {
    "base_url": "https://api.exchangerate-api.com/v4",
    "api_key": "",
    "timeout": 10,
    "retry_count": 3,
    "cache_timeout": 3600
  },
  "alpha_vantage": {
    "base_url": "https://www.alphavantage.co/query",
    "api_key": "",
    "timeout": 10,
    "retry_count": 3,
    "cache_timeout": 300
  },
  "system": {
    "use_ai": true,
    "use_real_data": true,
    "cache_enabled": true,
    "default_cache_timeout": 300,
    "log_api_calls": false
  }
}
```

## Próximos Passos

Após configurar:

1. ✅ Teste o sistema: `python start.py`
2. ✅ Verifique se APIs estão funcionando
3. ✅ Ajuste timeouts e limites conforme necessário
4. ✅ Configure cache conforme seu uso

