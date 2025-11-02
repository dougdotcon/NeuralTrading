# Configuração do Sistema

## Sistema de Configuração Centralizada

O sistema agora usa configuração centralizada na pasta `config/` para gerenciar todas as APIs.

### Arquivos de Configuração

- **`config/api_config.py`**: Módulo de gerenciamento de configurações
- **`config/api_keys.json`**: Arquivo com suas chaves de API (não commitado no Git)
- **`config/api_keys.json.example`**: Exemplo de configuração

### Como Configurar

1. **Copie o arquivo de exemplo:**
   ```bash
   cp config/api_keys.json.example config/api_keys.json
   ```

2. **Edite `config/api_keys.json`** e adicione suas chaves de API:
   ```json
   {
     "openrouter": {
       "api_key": "sk-or-v1-your-key-here"
     },
     "coingecko": {
       "api_key": "your-coingecko-key"
     }
   }
   ```

3. O sistema carregará automaticamente as configurações.

### APIs Disponíveis

#### OpenRouter / Deepseek

Para usar previsões de IA reais com Deepseek:

1. **Obtenha uma API Key:**
   - Acesse: https://openrouter.ai/keys
   - Crie uma conta (gratuita)
   - Gere uma API key

2. **Configure no arquivo `config/api_keys.json`:**

```json
{
  "openrouter": {
    "api_key": "sk-or-v1-your-key-here"
  }
}
```

**Alternativa: Variável de Ambiente**
Você ainda pode usar variáveis de ambiente (será usado como fallback se não houver no arquivo):

### Modelos Deepseek Disponíveis

- **deepseek/deepseek-r1-0528:free**: Gratuito, recomendado para testes
- **deepseek/deepseek-chat**: Pago, mais rápido
- **deepseek/deepseek-r1**: Pago, melhor qualidade

Por padrão, o sistema usa o modelo gratuito.

## Habilitar Previsões de IA

O sistema detecta automaticamente se `OPENROUTER_API_KEY` está configurada. Se estiver, usa Deepseek para previsões reais. Caso contrário, usa o modo padrão (simulação).

Para forçar uso de IA mesmo sem API key (apenas para testes), você pode modificar o código, mas não haverá previsões reais.

## Segurança

⚠️ **NUNCA** commite arquivos `.env` ou chaves de API no repositório Git!

O arquivo `.env` está no `.gitignore` por padrão.

