#!/usr/bin/env python3
"""
🔥 NEURAL TRADING CYBERPUNK TERMINAL 🔥
Interface Terminal Cyberpunk ASCII para Sistema de Trading com IA
Desenvolvido seguindo o padrão dos projetos ASIMOV
"""

import os
import sys
import time
import threading
from datetime import datetime
import colorama
from colorama import Fore, Back, Style
import pyfiglet

from .neural_config import (
    CYBERPUNK_COLORS, CYBERPUNK_SYMBOLS, POPULAR_ASSETS, TRADING_STRATEGIES,
    NEURAL_MODELS, get_timestamp, format_currency, format_percentage
)
from .neural_forecaster import NeuralForecaster, MultiModelForecaster
from .trading_strategies import StrategyManager
from .portfolio_manager import PortfolioManager

# Inicializar colorama para Windows
colorama.init()

class NeuralTradingTerminal:
    def __init__(self):
        self.running = True
        self.current_operation = None

        # Configuração de dados
        self.use_real_data = True  # Por padrão usa dados reais

        # Inicializar componentes
        self.forecaster = NeuralForecaster(use_real_data=self.use_real_data)
        self.multi_forecaster = MultiModelForecaster(use_real_data=self.use_real_data)
        self.strategy_manager = StrategyManager()
        self.portfolio_manager = PortfolioManager()

        # Criar portfólio padrão
        self.portfolio_manager.create_portfolio("Main Portfolio", 100000, "moderate")

        # Estado do sistema
        self.system_status = {
            'neural_engine': 'ONLINE',
            'gpu_acceleration': 'ENABLED',
            'trading_engine': 'ACTIVE',
            'risk_manager': 'MONITORING'
        }

    def clear_screen(self):
        """Limpa a tela"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def print_banner(self):
        """Exibe o banner cyberpunk ASCII"""
        self.clear_screen()

        # Banner principal
        banner = pyfiglet.figlet_format("NEURAL", font="slant")
        print(f"{Fore.CYAN}{Style.BRIGHT}{banner}{Style.RESET_ALL}")

        # Subtítulo cyberpunk
        print(f"{Fore.GREEN}{'═' * 80}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}    ▓▓▓ NEURAL TRADING AI TERMINAL v1.0 ▓▓▓{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'═' * 80}{Style.RESET_ALL}")

        # Arte ASCII cyberpunk
        ascii_art = """
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗                          ║
    ║  ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║                          ║
    ║  ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║                          ║
    ║  ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║                          ║
    ║  ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗                     ║
    ║  ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝                     ║
    ║                    AI-Powered Trading System                             ║
    ╚══════════════════════════════════════════════════════════════════════════╝
        """
        print(f"{Fore.MAGENTA}{ascii_art}{Style.RESET_ALL}")

        # Status do sistema
        data_mode = "REAL DATA" if self.use_real_data else "SIMULATED"
        data_color = Fore.GREEN if self.use_real_data else Fore.YELLOW

        print(f"{Fore.GREEN}[NEURAL ENGINE]{Style.RESET_ALL} {Fore.CYAN}{self.system_status['neural_engine']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[GPU ACCEL]{Style.RESET_ALL} {Fore.CYAN}{self.system_status['gpu_acceleration']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[TRADING]{Style.RESET_ALL} {Fore.CYAN}{self.system_status['trading_engine']}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[DATA MODE]{Style.RESET_ALL} {data_color}{data_mode}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[TIMESTAMP]{Style.RESET_ALL} {Fore.YELLOW}{get_timestamp()}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'═' * 80}{Style.RESET_ALL}")

    def print_menu(self):
        """Exibe o menu principal cyberpunk"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                           MENU PRINCIPAL                                 ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╠═══════════════════════════════════════════════════════════════════════════╣{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[1]{Style.RESET_ALL} {Fore.YELLOW}► PREVISÃO NEURAL{Style.RESET_ALL}                                          {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[2]{Style.RESET_ALL} {Fore.YELLOW}► ESTRATÉGIAS DE TRADING{Style.RESET_ALL}                                  {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[3]{Style.RESET_ALL} {Fore.YELLOW}► GERENCIAR PORTFÓLIO{Style.RESET_ALL}                                     {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[4]{Style.RESET_ALL} {Fore.YELLOW}► ANÁLISE DE RISCO{Style.RESET_ALL}                                        {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[5]{Style.RESET_ALL} {Fore.YELLOW}► BACKTESTING{Style.RESET_ALL}                                             {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[6]{Style.RESET_ALL} {Fore.YELLOW}► DASHBOARD TEMPO REAL{Style.RESET_ALL}                                    {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[7]{Style.RESET_ALL} {Fore.YELLOW}► CONFIGURAÇÕES{Style.RESET_ALL}                                           {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[8]{Style.RESET_ALL} {Fore.YELLOW}► ALTERNAR DADOS ({'REAIS' if self.use_real_data else 'SIMULADOS'}){Style.RESET_ALL}                        {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.RED}[0]{Style.RESET_ALL} {Fore.RED}► DESCONECTAR DO SISTEMA{Style.RESET_ALL}                                   {Fore.CYAN}║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

    def get_user_input(self, prompt, input_type="string"):
        """Obtém entrada do usuário com estilo cyberpunk"""
        while True:
            try:
                print(f"\n{Fore.GREEN}┌─[{Fore.CYAN}NEURAL{Fore.GREEN}]─[{Fore.YELLOW}INPUT{Fore.GREEN}]{Style.RESET_ALL}")
                user_input = input(f"{Fore.GREEN}└──╼ {Fore.CYAN}{prompt}{Style.RESET_ALL} {Fore.GREEN}►{Style.RESET_ALL} ")

                if input_type == "int":
                    return int(user_input)
                elif input_type == "choice":
                    if user_input in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                        return user_input
                    else:
                        self.print_error("Opção inválida! Digite 0-8.")
                        continue
                else:
                    return user_input.strip()
            except ValueError:
                self.print_error(f"Entrada inválida! Digite um {input_type} válido.")
            except KeyboardInterrupt:
                self.print_warning("\nOperação cancelada pelo usuário.")
                return None

    def print_loading_animation(self, message, duration=3):
        """Exibe animação de carregamento cyberpunk"""
        frames = ['▓', '▒', '░', '▒']

        for i in range(duration * 4):
            frame = frames[i % len(frames)]
            print(f"\r{Fore.YELLOW}[{frame}{frame}{frame}] {message} [{frame}{frame}{frame}]{Style.RESET_ALL}", end='', flush=True)
            time.sleep(0.25)
        print()

    def print_success(self, message):
        """Exibe mensagem de sucesso"""
        print(f"{Fore.GREEN}[✓ SUCESSO]{Style.RESET_ALL} {message}")

    def print_error(self, message):
        """Exibe mensagem de erro"""
        print(f"{Fore.RED}[✗ ERRO]{Style.RESET_ALL} {message}")

    def print_warning(self, message):
        """Exibe mensagem de aviso"""
        print(f"{Fore.YELLOW}[⚠ AVISO]{Style.RESET_ALL} {message}")

    def print_info(self, message):
        """Exibe mensagem informativa"""
        print(f"{Fore.CYAN}[ℹ INFO]{Style.RESET_ALL} {message}")

    def neural_forecast_menu(self):
        """Menu de previsão neural"""
        while True:
            self.print_banner()
            print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║                         PREVISÃO NEURAL                                  ║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}╠═══════════════════════════════════════════════════════════════════════════╣{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[1]{Style.RESET_ALL} {Fore.YELLOW}► Previsão Individual{Style.RESET_ALL}                                      {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[2]{Style.RESET_ALL} {Fore.YELLOW}► Previsão em Lote{Style.RESET_ALL}                                         {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[3]{Style.RESET_ALL} {Fore.YELLOW}► Previsão Ensemble{Style.RESET_ALL}                                        {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[4]{Style.RESET_ALL} {Fore.YELLOW}► Trocar Modelo Neural{Style.RESET_ALL}                                     {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[5]{Style.RESET_ALL} {Fore.YELLOW}► Performance do Modelo{Style.RESET_ALL}                                    {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.RED}[0]{Style.RESET_ALL} {Fore.RED}► Voltar ao Menu Principal{Style.RESET_ALL}                                  {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

            choice = self.get_user_input("Escolha uma opção", "choice")

            if choice == "0":
                break
            elif choice == "1":
                self.single_prediction()
            elif choice == "2":
                self.batch_prediction()
            elif choice == "3":
                self.ensemble_prediction()
            elif choice == "4":
                self.switch_neural_model()
            elif choice == "5":
                self.show_model_performance()

    def single_prediction(self):
        """Realiza previsão individual"""
        self.print_info("=== PREVISÃO NEURAL INDIVIDUAL ===")

        # Mostra ativos populares
        print(f"\n{Fore.YELLOW}📈 ATIVOS POPULARES:{Style.RESET_ALL}")
        for category, assets in POPULAR_ASSETS.items():
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} {category.upper()}: {', '.join(assets[:4])}")

        symbol = self.get_user_input("Digite o símbolo do ativo (ex: AAPL, BTC)")
        if not symbol:
            return

        horizon = self.get_user_input("Horizonte de previsão (horas, padrão: 24)", "string")
        try:
            horizon = int(horizon) if horizon else 24
        except:
            horizon = 24

        self.print_loading_animation(f"Gerando previsão neural para {symbol.upper()}")

        try:
            prediction = self.forecaster.predict(symbol.upper(), horizon)
            self.display_prediction_result(prediction)
        except Exception as e:
            self.print_error(f"Erro na previsão: {str(e)}")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def display_prediction_result(self, prediction):
        """Exibe resultado da previsão"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                      RESULTADO DA PREVISÃO                               ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}🎯 ATIVO:{Style.RESET_ALL} {prediction['symbol']}")
        print(f"{Fore.YELLOW}🤖 MODELO:{Style.RESET_ALL} {prediction['model'].upper()}")
        print(f"{Fore.YELLOW}📊 HORIZONTE:{Style.RESET_ALL} {prediction['horizon']} períodos")
        print(f"{Fore.YELLOW}💰 PREÇO ATUAL:{Style.RESET_ALL} {format_currency(prediction['current_price'])}")

        # Previsão final
        final_price = prediction['predictions'][-1]
        price_change = (final_price - prediction['current_price']) / prediction['current_price'] * 100

        color = Fore.GREEN if price_change > 0 else Fore.RED
        print(f"{Fore.YELLOW}🔮 PREVISÃO FINAL:{Style.RESET_ALL} {format_currency(final_price)} ({color}{format_percentage(price_change)}{Style.RESET_ALL})")

        # Métricas
        metrics = prediction['metrics']
        print(f"\n{Fore.YELLOW}📈 MÉTRICAS DO MODELO:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Acurácia: {format_percentage(metrics['accuracy'])}")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} R² Score: {metrics['r2_score']:.3f}")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} MAPE: {metrics['mape']:.2f}%")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Tempo de Inferência: {metrics['inference_time_ms']:.1f}ms")

        # Sinal de trading
        signal = self.forecaster.get_signal_strength(prediction)
        signal_color = Fore.GREEN if signal['direction'] == 'Bullish' else Fore.RED
        print(f"\n{Fore.YELLOW}⚡ SINAL DE TRADING:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Direção: {signal_color}{signal['direction']}{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Força: {signal['strength']}")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Confiança: {format_percentage(signal['confidence'])}")

    def trading_strategies_menu(self):
        """Menu de estratégias de trading"""
        while True:
            self.print_banner()
            print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║                      ESTRATÉGIAS DE TRADING                               ║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}╠═══════════════════════════════════════════════════════════════════════════╣{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[1]{Style.RESET_ALL} {Fore.YELLOW}► Executar Estratégia{Style.RESET_ALL}                                      {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[2]{Style.RESET_ALL} {Fore.YELLOW}► Comparar Estratégias{Style.RESET_ALL}                                     {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[3]{Style.RESET_ALL} {Fore.YELLOW}► Configurar Estratégia{Style.RESET_ALL}                                    {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.GREEN}[4]{Style.RESET_ALL} {Fore.YELLOW}► Ver Estratégias Disponíveis{Style.RESET_ALL}                             {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{Style.RESET_ALL} {Fore.RED}[0]{Style.RESET_ALL} {Fore.RED}► Voltar ao Menu Principal{Style.RESET_ALL}                                  {Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

            choice = self.get_user_input("Escolha uma opção", "choice")

            if choice == "0":
                break
            elif choice == "1":
                self.execute_strategy()
            elif choice == "2":
                self.compare_strategies()
            elif choice == "3":
                self.print_info("Configuração de estratégias em desenvolvimento...")
                input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")
            elif choice == "4":
                self.show_available_strategies()

    def show_available_strategies(self):
        """Mostra estratégias disponíveis"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                     ESTRATÉGIAS DISPONÍVEIS                              ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        for strategy_id, info in TRADING_STRATEGIES.items():
            risk_color = Fore.GREEN if info['risk_level'] == 'Low' else Fore.YELLOW if info['risk_level'] == 'Medium' else Fore.RED

            print(f"\n{Fore.YELLOW}🎯 {info['name'].upper()}{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} Descrição: {info['description']}")
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} Risco: {risk_color}{info['risk_level']}{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} Timeframe: {info['timeframe']}")
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} Sharpe Target: {info['sharpe_target']}")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def batch_prediction(self):
        """Realiza previsão em lote"""
        self.print_info("=== PREVISÃO NEURAL EM LOTE ===")

        print(f"\n{Fore.YELLOW}📈 SELECIONE CATEGORIA:{Style.RESET_ALL}")
        categories = list(POPULAR_ASSETS.keys())
        for i, category in enumerate(categories, 1):
            print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {category.upper()}")

        try:
            choice = int(self.get_user_input("Escolha a categoria (1-4)")) - 1
            if 0 <= choice < len(categories):
                category = categories[choice]
                symbols = POPULAR_ASSETS[category][:4]  # Primeiros 4 ativos

                self.print_loading_animation(f"Executando previsão em lote para {category}")

                batch_result = self.forecaster.batch_predict(symbols)
                self.display_batch_results(batch_result)
            else:
                self.print_error("Categoria inválida!")
        except:
            self.print_error("Entrada inválida!")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def display_batch_results(self, batch_result):
        """Exibe resultados da previsão em lote"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                      RESULTADOS EM LOTE                                  ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        metrics = batch_result['batch_metrics']
        print(f"\n{Fore.YELLOW}⚡ MÉTRICAS DO LOTE:{Style.RESET_ALL}")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Tempo Total: {metrics['total_time']:.2f}s")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Throughput: {metrics['throughput']:.1f} previsões/s")
        print(f"  {Fore.GREEN}►{Style.RESET_ALL} Tempo Médio: {metrics['avg_time_per_symbol']:.2f}s por ativo")

        print(f"\n{Fore.YELLOW}📊 RESULTADOS POR ATIVO:{Style.RESET_ALL}")
        for symbol, result in batch_result['results'].items():
            current_price = result['current_price']
            final_price = result['predictions'][-1]
            change_pct = (final_price - current_price) / current_price * 100

            color = Fore.GREEN if change_pct > 0 else Fore.RED
            print(f"  {Fore.CYAN}{symbol}{Style.RESET_ALL}: {format_currency(current_price)} → {format_currency(final_price)} ({color}{format_percentage(change_pct)}{Style.RESET_ALL})")

    def ensemble_prediction(self):
        """Realiza previsão ensemble"""
        self.print_info("=== PREVISÃO ENSEMBLE (MÚLTIPLOS MODELOS) ===")

        symbol = self.get_user_input("Digite o símbolo do ativo")
        if not symbol:
            return

        self.print_loading_animation(f"Executando previsão ensemble para {symbol.upper()}")

        try:
            ensemble_result = self.multi_forecaster.ensemble_predict(symbol.upper())
            self.display_ensemble_results(ensemble_result)
        except Exception as e:
            self.print_error(f"Erro na previsão ensemble: {str(e)}")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def display_ensemble_results(self, ensemble_result):
        """Exibe resultados da previsão ensemble"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                      PREVISÃO ENSEMBLE                                   ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}🎭 ATIVO:{Style.RESET_ALL} {ensemble_result['symbol']}")

        # Mostra pesos dos modelos
        print(f"\n{Fore.YELLOW}⚖️ PESOS DOS MODELOS:{Style.RESET_ALL}")
        for model, weight in ensemble_result['weights'].items():
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} {model.upper()}: {format_percentage(weight * 100)}")

        # Mostra previsões individuais vs ensemble
        print(f"\n{Fore.YELLOW}🔮 PREVISÕES FINAIS:{Style.RESET_ALL}")
        for model, prediction in ensemble_result['individual_predictions'].items():
            final_price = prediction['predictions'][-1]
            print(f"  {Fore.CYAN}{model.upper()}{Style.RESET_ALL}: {format_currency(final_price)}")

        ensemble_final = ensemble_result['ensemble_predictions'][-1]
        print(f"  {Fore.YELLOW}ENSEMBLE{Style.RESET_ALL}: {format_currency(ensemble_final)}")

    def switch_neural_model(self):
        """Troca modelo neural"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                       MODELOS NEURAIS                                    ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}🤖 MODELO ATUAL:{Style.RESET_ALL} {self.forecaster.model_type.upper()}")

        print(f"\n{Fore.YELLOW}📋 MODELOS DISPONÍVEIS:{Style.RESET_ALL}")
        models = list(NEURAL_MODELS.keys())
        for i, model in enumerate(models, 1):
            info = NEURAL_MODELS[model]
            current = " (ATUAL)" if model == self.forecaster.model_type else ""
            print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {info['name']} - {info['accuracy']} acurácia{current}")

        try:
            choice = int(self.get_user_input("Escolha o modelo (1-3)")) - 1
            if 0 <= choice < len(models):
                new_model = models[choice]
                if self.forecaster.switch_model(new_model):
                    self.print_success(f"Modelo alterado para {new_model.upper()}")
                else:
                    self.print_error("Erro ao alterar modelo")
            else:
                self.print_error("Modelo inválido!")
        except:
            self.print_error("Entrada inválida!")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def show_model_performance(self):
        """Mostra performance do modelo"""
        performance = self.forecaster.get_model_performance()

        if not performance:
            self.print_warning("Nenhuma previsão realizada ainda.")
            input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")
            return

        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                    PERFORMANCE DO MODELO                                 ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}🤖 MODELO:{Style.RESET_ALL} {performance['model_type'].upper()}")
        print(f"{Fore.YELLOW}📊 TOTAL DE PREVISÕES:{Style.RESET_ALL} {performance['total_predictions']}")
        print(f"{Fore.YELLOW}🎯 ACURÁCIA MÉDIA:{Style.RESET_ALL} {format_percentage(performance['avg_accuracy'])}")
        print(f"{Fore.YELLOW}⚡ TEMPO MÉDIO:{Style.RESET_ALL} {performance['avg_inference_time_ms']:.1f}ms")
        print(f"{Fore.YELLOW}🚀 GPU ATIVADO:{Style.RESET_ALL} {'Sim' if performance['gpu_enabled'] else 'Não'}")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def execute_strategy(self):
        """Executa estratégia de trading"""
        self.print_info("=== EXECUTAR ESTRATÉGIA ===")

        print(f"\n{Fore.YELLOW}🎯 ESTRATÉGIAS DISPONÍVEIS:{Style.RESET_ALL}")
        strategies = list(TRADING_STRATEGIES.keys())
        for i, strategy in enumerate(strategies, 1):
            info = TRADING_STRATEGIES[strategy]
            print(f"  {Fore.GREEN}[{i}]{Style.RESET_ALL} {info['name']}")

        try:
            choice = int(self.get_user_input("Escolha a estratégia (1-4)")) - 1
            if 0 <= choice < len(strategies):
                strategy_type = strategies[choice]

                symbol = self.get_user_input("Digite o símbolo do ativo")
                if not symbol:
                    return

                self.print_loading_animation(f"Executando estratégia {strategy_type} para {symbol.upper()}")

                strategy = self.strategy_manager.get_strategy(strategy_type)
                market_analysis = strategy.analyze_market(symbol.upper())
                signal = strategy.generate_signal(market_analysis)

                self.display_trading_signal(signal, strategy_type)
            else:
                self.print_error("Estratégia inválida!")
        except:
            self.print_error("Entrada inválida!")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def display_trading_signal(self, signal, strategy_type):
        """Exibe sinal de trading"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                        SINAL DE TRADING                                  ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        action_color = Fore.GREEN if signal['action'] == 'BUY' else Fore.RED if signal['action'] == 'SELL' else Fore.YELLOW

        print(f"\n{Fore.YELLOW}🎯 ESTRATÉGIA:{Style.RESET_ALL} {TRADING_STRATEGIES[strategy_type]['name']}")
        print(f"{Fore.YELLOW}📈 ATIVO:{Style.RESET_ALL} {signal['symbol']}")
        print(f"{Fore.YELLOW}⚡ AÇÃO:{Style.RESET_ALL} {action_color}{signal['action']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💰 PREÇO ENTRADA:{Style.RESET_ALL} {format_currency(signal['entry_price'])}")

        if signal['stop_loss']:
            print(f"{Fore.YELLOW}🛑 STOP LOSS:{Style.RESET_ALL} {format_currency(signal['stop_loss'])}")
        if signal['take_profit']:
            print(f"{Fore.YELLOW}🎯 TAKE PROFIT:{Style.RESET_ALL} {format_currency(signal['take_profit'])}")

        print(f"{Fore.YELLOW}🎲 CONFIANÇA:{Style.RESET_ALL} {format_percentage(signal['confidence'] * 100)}")
        print(f"{Fore.YELLOW}📝 RAZÃO:{Style.RESET_ALL} {signal['reasoning']}")
        print(f"{Fore.YELLOW}📊 RETORNO ESPERADO:{Style.RESET_ALL} {format_percentage(signal['expected_return'])}")

    def compare_strategies(self):
        """Compara estratégias"""
        self.print_info("=== COMPARAÇÃO DE ESTRATÉGIAS ===")

        symbol = self.get_user_input("Digite o símbolo do ativo para comparação")
        if not symbol:
            return

        self.print_loading_animation(f"Comparando todas as estratégias para {symbol.upper()}")

        try:
            comparison = self.strategy_manager.run_strategy_comparison(symbol.upper())
            self.display_strategy_comparison(comparison)
        except Exception as e:
            self.print_error(f"Erro na comparação: {str(e)}")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def display_strategy_comparison(self, comparison):
        """Exibe comparação de estratégias"""
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.CYAN}║                    COMPARAÇÃO DE ESTRATÉGIAS                             ║{Style.RESET_ALL}")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}📈 ATIVO:{Style.RESET_ALL} {comparison['symbol']}")
        print(f"{Fore.YELLOW}🏆 MELHOR ESTRATÉGIA:{Style.RESET_ALL} {comparison['best_strategy'].upper()}")

        print(f"\n{Fore.YELLOW}📊 RESULTADOS POR ESTRATÉGIA:{Style.RESET_ALL}")

        for strategy_name, result in comparison['results'].items():
            signal = result['signal']
            action_color = Fore.GREEN if signal['action'] == 'BUY' else Fore.RED if signal['action'] == 'SELL' else Fore.YELLOW

            print(f"\n  {Fore.CYAN}🎯 {strategy_name.upper()}{Style.RESET_ALL}")
            print(f"    {Fore.GREEN}►{Style.RESET_ALL} Ação: {action_color}{signal['action']}{Style.RESET_ALL}")
            print(f"    {Fore.GREEN}►{Style.RESET_ALL} Confiança: {format_percentage(signal['confidence'] * 100)}")
            print(f"    {Fore.GREEN}►{Style.RESET_ALL} Retorno Esperado: {format_percentage(signal['expected_return'])}")
            print(f"    {Fore.GREEN}►{Style.RESET_ALL} Sharpe Target: {result['expected_sharpe']}")

    def toggle_data_source(self):
        """Alterna entre dados reais e simulados"""
        self.print_info("=== ALTERNAR FONTE DE DADOS ===")

        current_mode = "DADOS REAIS" if self.use_real_data else "DADOS SIMULADOS"
        new_mode = "DADOS SIMULADOS" if self.use_real_data else "DADOS REAIS"

        print(f"\n{Fore.YELLOW}📊 MODO ATUAL:{Style.RESET_ALL} {current_mode}")
        print(f"{Fore.YELLOW}🔄 NOVO MODO:{Style.RESET_ALL} {new_mode}")

        if self.use_real_data:
            print(f"\n{Fore.CYAN}ℹ️ SOBRE DADOS SIMULADOS:{Style.RESET_ALL}")
            print("  • Dados gerados algoritmicamente")
            print("  • Sempre disponíveis")
            print("  • Padrões consistentes para testes")
            print("  • Sem dependência de APIs externas")
        else:
            print(f"\n{Fore.CYAN}ℹ️ SOBRE DADOS REAIS:{Style.RESET_ALL}")
            print("  • Dados de mercado em tempo real")
            print("  • APIs gratuitas (Yahoo Finance, CoinGecko)")
            print("  • Indicadores técnicos reais")
            print("  • Preços atualizados do mercado")

        confirm = self.get_user_input("Confirma a alteração? (s/n)")

        if confirm and confirm.lower() in ['s', 'sim', 'y', 'yes']:
            self.print_loading_animation("Alterando fonte de dados")

            # Alterna o modo
            self.use_real_data = not self.use_real_data

            # Reinicializa componentes
            self.forecaster = NeuralForecaster(use_real_data=self.use_real_data)
            self.multi_forecaster = MultiModelForecaster(use_real_data=self.use_real_data)

            new_mode_final = "DADOS REAIS" if self.use_real_data else "DADOS SIMULADOS"
            self.print_success(f"Fonte alterada para: {new_mode_final}")

            # Testa conectividade se mudou para dados reais
            if self.use_real_data:
                print(f"\n{Fore.YELLOW}🔍 Testando conectividade com APIs...{Style.RESET_ALL}")
                try:
                    test_results = self.forecaster.data_collector.test_apis()
                    success_count = sum(test_results.values())
                    total_count = len(test_results)

                    if success_count == total_count:
                        self.print_success("Todas as APIs estão funcionando!")
                    elif success_count > 0:
                        self.print_warning(f"{success_count}/{total_count} APIs funcionando")
                    else:
                        self.print_error("Nenhuma API funcionando - usando fallback simulado")
                except Exception as e:
                    self.print_warning(f"Erro no teste de APIs: {str(e)}")
        else:
            self.print_info("Alteração cancelada")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def test_real_data_apis(self):
        """Testa conectividade com APIs de dados reais"""
        if not self.use_real_data:
            self.print_warning("Sistema está em modo de dados simulados")
            input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")
            return

        self.print_info("=== TESTE DE CONECTIVIDADE COM APIS ===")

        try:
            self.print_loading_animation("Testando APIs de dados reais")

            test_results = self.forecaster.data_collector.test_apis()

            print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║                      RESULTADOS DOS TESTES                               ║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}")

            for symbol, success in test_results.items():
                status_color = Fore.GREEN if success else Fore.RED
                status_text = "✅ SUCESSO" if success else "❌ FALHA"
                print(f"  {Fore.YELLOW}{symbol}{Style.RESET_ALL}: {status_color}{status_text}{Style.RESET_ALL}")

            success_count = sum(test_results.values())
            total_count = len(test_results)
            success_rate = (success_count / total_count) * 100

            print(f"\n{Fore.YELLOW}📊 RESUMO:{Style.RESET_ALL}")
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} APIs funcionando: {success_count}/{total_count}")
            print(f"  {Fore.GREEN}►{Style.RESET_ALL} Taxa de sucesso: {success_rate:.1f}%")

            if success_rate >= 75:
                self.print_success("Conectividade excelente!")
            elif success_rate >= 50:
                self.print_warning("Conectividade moderada")
            else:
                self.print_error("Conectividade baixa - considere usar dados simulados")

        except Exception as e:
            self.print_error(f"Erro no teste de APIs: {str(e)}")

        input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

    def run(self):
        """Loop principal da interface"""
        while self.running:
            try:
                self.print_banner()
                self.print_menu()

                choice = self.get_user_input("Selecione uma opção", "choice")

                if choice == "0":
                    self.print_loading_animation("Desconectando do sistema neural", 2)
                    self.print_success("Sistema desconectado com segurança!")
                    self.running = False

                elif choice == "1":
                    self.neural_forecast_menu()

                elif choice == "2":
                    self.trading_strategies_menu()

                elif choice == "3":
                    self.print_info("Gerenciamento de portfólio em desenvolvimento...")
                    input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

                elif choice == "4":
                    self.print_info("Análise de risco em desenvolvimento...")
                    input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

                elif choice == "5":
                    self.print_info("Backtesting em desenvolvimento...")
                    input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

                elif choice == "6":
                    self.print_info("Dashboard tempo real em desenvolvimento...")
                    input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

                elif choice == "7":
                    self.print_info("Configurações em desenvolvimento...")
                    input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

                elif choice == "8":
                    self.toggle_data_source()

                elif choice is None:
                    self.running = False

            except KeyboardInterrupt:
                self.print_warning("\nSaindo do sistema...")
                self.running = False
            except Exception as e:
                self.print_error(f"Erro inesperado: {str(e)}")
                input(f"\n{Fore.YELLOW}Pressione ENTER para continuar...{Style.RESET_ALL}")

def main():
    """Função principal"""
    try:
        terminal = NeuralTradingTerminal()
        terminal.run()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Sistema interrompido pelo usuário.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}Erro inesperado: {str(e)}{Style.RESET_ALL}")
    finally:
        print(f"{Fore.CYAN}Obrigado por usar o NeuralTrading!{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
