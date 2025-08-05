#!/usr/bin/env python3
"""
🔥 NEURAL TRADING CYBERPUNK LAUNCHER 🔥
Launcher automático com verificação e instalação de dependências
"""

import os
import sys
import subprocess
import importlib.util

def check_python_version():
    """Verifica se a versão do Python é compatível"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ é necessário!")
        print(f"Versão atual: {sys.version}")
        return False
    return True

def check_package(package_name):
    """Verifica se um pacote está instalado"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Instala um pacote usando pip"""
    try:
        print(f"📦 Instalando {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Erro ao instalar {package_name}")
        return False

def check_and_install_dependencies():
    """Verifica e instala dependências necessárias"""
    dependencies = [
        "colorama",
        "pyfiglet",
        "numpy",
        "pandas",
        "requests"
    ]

    print("🔍 Verificando dependências...")

    missing_packages = []
    for package in dependencies:
        if not check_package(package):
            missing_packages.append(package)

    if missing_packages:
        print(f"📋 Pacotes faltando: {', '.join(missing_packages)}")

        response = input("Deseja instalar automaticamente? (s/n): ").lower()
        if response in ['s', 'sim', 'y', 'yes']:
            for package in missing_packages:
                if not install_package(package):
                    return False
            print("✅ Todas as dependências foram instaladas!")
        else:
            print("❌ Instalação cancelada pelo usuário.")
            return False
    else:
        print("✅ Todas as dependências estão instaladas!")

    return True

def create_requirements_file():
    """Cria arquivo requirements.txt se não existir"""
    requirements_content = """colorama>=0.4.4
pyfiglet>=0.8.post1
numpy>=1.21.0
pandas>=1.3.0
requests>=2.25.0
"""

    if not os.path.exists("requirements.txt"):
        with open("requirements.txt", "w") as f:
            f.write(requirements_content)
        print("📄 Arquivo requirements.txt criado!")

def show_startup_banner():
    """Exibe banner de inicialização"""
    print("🚀 NEURAL TRADING CYBERPUNK LAUNCHER")
    print("=" * 50)
    print("🤖 Sistema de Trading com IA Neural")
    print("⚡ GPU Acceleration Ready")
    print("🎯 Estratégias Avançadas")
    print("=" * 50)

def main():
    """Função principal do launcher"""
    show_startup_banner()

    # Verificar versão do Python
    if not check_python_version():
        input("Pressione ENTER para sair...")
        return 1

    # Criar arquivo requirements.txt
    create_requirements_file()

    # Verificar e instalar dependências
    if not check_and_install_dependencies():
        print("❌ Erro na instalação de dependências!")
        input("Pressione ENTER para sair...")
        return 1

    print("\n🎯 Iniciando interface cyberpunk do NeuralTrading...")

    try:
        # Importar e executar a interface cyberpunk
        from cyberpunk_neural_terminal import main as cyberpunk_main
        print("🚀 Carregando NEURAL TRADING AI...")
        cyberpunk_main()
    except ImportError as e:
        print(f"❌ Erro ao importar interface cyberpunk: {e}")
        print("Verifique se o arquivo cyberpunk_neural_terminal.py existe.")
        input("Pressione ENTER para sair...")
        return 1
    except KeyboardInterrupt:
        print("\n\n👋 Sistema encerrado pelo usuário.")
        return 0
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        print("📧 Reporte este erro para suporte técnico.")
        input("Pressione ENTER para sair...")
        return 1

    print("\n👋 Obrigado por usar NEURAL TRADING AI!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
