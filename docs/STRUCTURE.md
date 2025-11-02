# Estrutura do Projeto Neural Trading

## Organização de Arquivos e Pastas

Este documento descreve a estrutura organizada do projeto Neural Trading Cyberpunk Terminal.

## Estrutura de Diretórios

```
TRADING_neural/
│
├── src/                              # Código fonte do sistema
│   └── neural_trading/               # Pacote principal
│       ├── __init__.py               # Inicialização do pacote
│       ├── cyberpunk_neural_terminal.py
│       ├── neural_config.py
│       ├── neural_forecaster.py
│       ├── real_data_collector.py
│       ├── trading_strategies.py
│       └── portfolio_manager.py
│
├── scripts/                          # Scripts de inicialização
│   ├── start_neural_cyberpunk.py     # Launcher principal Python
│   └── start_neural_cyberpunk.bat    # Launcher Windows
│
├── docs/                             # Documentação
│   ├── DEMO.md
│   ├── performance.md
│   ├── PROJECT_SUMMARY.md
│   ├── README_UPDATE_SUMMARY.md
│   ├── REAL_DATA_DEMO.md
│   ├── REAL_DATA_UPDATE.md
│   ├── STRUCTURE.md                  # Este arquivo
│   └── ...
│
├── logs/                             # Logs do sistema
│   ├── combined.log
│   ├── error.log
│   └── interactions.log
│
├── tests/                            # Testes unitários (a implementar)
│
├── config/                           # Configurações adicionais (futuro)
│
├── start.bat                         # Launcher principal Windows (raiz)
├── start.py                          # Launcher principal Python (raiz)
├── requirements.txt                  # Dependências do projeto
├── README.md                         # Documentação principal
└── .gitignore                        # Arquivos ignorados pelo Git
```

## Descrição das Pastas

### `src/neural_trading/`
Contém todo o código fonte do sistema organizado como um pacote Python. Todos os módulos principais estão aqui:

- **cyberpunk_neural_terminal.py**: Interface principal do terminal
- **neural_config.py**: Configurações, constantes e funções auxiliares
- **neural_forecaster.py**: Engine de previsão neural
- **real_data_collector.py**: Coletor de dados reais de mercado
- **trading_strategies.py**: Implementação das estratégias de trading
- **portfolio_manager.py**: Gerenciamento de portfólio e risco

### `scripts/`
Scripts de inicialização e launchers:

- **start_neural_cyberpunk.py**: Launcher principal com verificação de dependências
- **start_neural_cyberpunk.bat**: Launcher Windows (legado, mantido para compatibilidade)

### `docs/`
Documentação completa do projeto:

- Documentação de funcionalidades
- Guias de uso
- Demonstrações
- Especificações

### `logs/`
Logs gerados pelo sistema durante execução.

### `tests/`
Diretório reservado para testes unitários e de integração (a implementar).

### `config/`
Diretório reservado para arquivos de configuração adicionais (futuro).

## Imports e Dependências

Todos os módulos em `src/neural_trading/` usam imports relativos:

```python
from .neural_config import CYBERPUNK_COLORS
from .neural_forecaster import NeuralForecaster
```

O script de inicialização (`scripts/start_neural_cyberpunk.py`) adiciona o diretório `src/` ao `sys.path`, permitindo que o pacote seja importado corretamente.

## Como Executar

### Opção 1: Launcher na Raiz (Recomendado)
```bash
# Windows
start.bat

# Linux/Mac
python start.py
```

### Opção 2: Launcher Direto
```bash
# Windows
scripts\start_neural_cyberpunk.bat

# Linux/Mac
python scripts/start_neural_cyberpunk.py
```

### Opção 3: Execução Manual
```bash
pip install -r requirements.txt
python -m neural_trading.cyberpunk_neural_terminal
```

## Vantagens da Nova Estrutura

1. **Organização Clara**: Código fonte separado de scripts e documentação
2. **Pacote Python**: Estrutura que segue convenções do Python
3. **Escalabilidade**: Fácil adicionar novos módulos ou funcionalidades
4. **Manutenibilidade**: Código mais fácil de entender e manter
5. **Testabilidade**: Estrutura preparada para testes
6. **Profissionalismo**: Segue boas práticas de desenvolvimento Python

## Migração da Estrutura Anterior

A estrutura anterior tinha todos os arquivos Python na raiz do projeto. A nova estrutura:

- Move módulos para `src/neural_trading/`
- Move scripts para `scripts/`
- Mantém documentação em `docs/`
- Adiciona launchers na raiz para facilitar execução
- Atualiza todos os imports para usar imports relativos

## Próximos Passos

- [ ] Implementar testes em `tests/`
- [ ] Adicionar arquivos de configuração em `config/`
- [ ] Melhorar organização de logs
- [ ] Considerar estrutura para dados persistentes

