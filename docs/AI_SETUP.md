# Guia Rápido: Configurar Deepseek IA

## Passo a Passo (5 minutos)

### 1. Obter API Key Gratuita

1. Acesse: https://openrouter.ai/keys
2. Clique em "Sign Up" ou "Login"
3. Após login, clique em "Create Key"
4. Copie a chave (formato: `sk-or-v1-...`)

### 2. Configurar no Windows

#### Opção A: Variável de Ambiente (Temporária)
Abra PowerShell e execute:
```powershell
$env:OPENROUTER_API_KEY="sua-chave-aqui"
```

#### Opção B: Arquivo .env (Permanente - Recomendado)
1. Na raiz do projeto, crie arquivo `.env`
2. Adicione:
```
OPENROUTER_API_KEY=sua-chave-aqui
```

### 3. Instalar Dependências

```bash
pip install openai python-dotenv
```

Ou:
```bash
pip install -r requirements.txt
```

### 4. Testar

Execute o sistema:
```bash
python start.py
```

Se tudo estiver OK, você verá:
```
🤖 Modo IA Deepseek ativado
```

No menu principal, aparece:
```
IA Deepseek: ATIVADO
```

### 5. Usar

1. Vá para "Previsão Neural" (opção 1)
2. Escolha "Previsão Individual" (opção 1)
3. Digite o símbolo (ex: AAPL)
4. O sistema usará Deepseek automaticamente!

## Verificar se Funcionou

Se a IA estiver ativada, você verá nas previsões:
- ✅ Mensagem "🤖 Usando Deepseek IA..."
- ✅ Seção "📈 INSIGHTS DA IA" com:
  - Tendência
  - Força do sinal
  - Suporte/Resistência
  - Raciocínio da IA

## Troubleshooting

### Não aparece "IA Deepseek ativado"
- Verifique se a variável está configurada:
  ```powershell
  echo $env:OPENROUTER_API_KEY
  ```
- Reinicie o terminal após configurar

### Erro de API
- Verifique se a chave está correta
- Verifique conexão com internet
- Verifique se tem créditos no OpenRouter

### Usa modo padrão mesmo com chave
- Verifique logs de erro no console
- Tente configurar novamente
- Verifique se `openai` está instalado

## Modelo Gratuito

O sistema usa por padrão:
- `deepseek/deepseek-r1-0528:free` (gratuito)

Você pode usar sem custos para testes!

