#!/usr/bin/env python3
"""
🔥 NEURAL TRADING CYBERPUNK LAUNCHER 🔥
Ponto de entrada principal na raiz do projeto
"""

import os
import sys

# Obter diretório do script atual
project_root = os.path.dirname(os.path.abspath(__file__))
scripts_path = os.path.join(project_root, "scripts")

# Adicionar scripts ao path
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)

# Importar e executar o launcher principal
try:
    from start_neural_cyberpunk import main
    sys.exit(main())
except ImportError as e:
    print(f"❌ Erro ao importar launcher: {e}")
    print(f"Verifique se o arquivo existe em: {scripts_path}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erro inesperado: {e}")
    sys.exit(1)

