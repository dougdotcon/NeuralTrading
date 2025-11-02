#!/usr/bin/env python3
"""
Configuração centralizada de APIs
Gerencia todas as chaves de API e configurações do sistema
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Any


class APIConfig:
    """Gerenciador centralizado de configurações de API"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa configurações de API
        
        Args:
            config_file: Caminho para arquivo de configuração (opcional)
        """
        # Determinar caminho do arquivo de configuração
        if config_file is None:
            # Tenta encontrar config.json na pasta config
            project_root = Path(__file__).parent.parent
            config_file = project_root / "config" / "api_keys.json"
        else:
            config_file = Path(config_file)
        
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configurações do arquivo ou cria padrão"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Erro ao carregar config: {e}")
                return self._default_config()
        else:
            # Criar arquivo de configuração padrão
            default = self._default_config()
            self._save_config(default)
            return default
    
    def _default_config(self) -> Dict[str, Any]:
        """Retorna configuração padrão"""
        return {
            "openrouter": {
                "api_key": os.getenv("OPENROUTER_API_KEY", ""),
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
                "api_key": os.getenv("COINGECKO_API_KEY", ""),
                "timeout": 10,
                "retry_count": 3,
                "cache_timeout": 300,
                "rate_limit": 50  # requests per minute (free tier)
            },
            "exchangerate": {
                "base_url": "https://api.exchangerate-api.com/v4",
                "api_key": os.getenv("EXCHANGERATE_API_KEY", ""),
                "timeout": 10,
                "retry_count": 3,
                "cache_timeout": 3600  # 1 hora (dados de forex mudam menos)
            },
            "alpha_vantage": {
                "base_url": "https://www.alphavantage.co/query",
                "api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
                "timeout": 10,
                "retry_count": 3,
                "cache_timeout": 300
            },
            "system": {
                "use_ai": True,
                "use_real_data": True,
                "cache_enabled": True,
                "default_cache_timeout": 300,
                "log_api_calls": False
            }
        }
    
    def _save_config(self, config: Dict[str, Any]):
        """Salva configuração no arquivo"""
        try:
            # Garantir que o diretório existe
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Erro ao salvar config: {e}")
    
    def get(self, service: str, key: str = None, default: Any = None) -> Any:
        """
        Obtém configuração de um serviço
        
        Args:
            service: Nome do serviço (openrouter, yahoo_finance, etc.)
            key: Chave específica (opcional)
            default: Valor padrão se não encontrado
            
        Returns:
            Valor da configuração
        """
        if service not in self.config:
            return default
        
        if key is None:
            return self.config[service]
        
        return self.config[service].get(key, default)
    
    def set(self, service: str, key: str, value: Any):
        """
        Define configuração de um serviço
        
        Args:
            service: Nome do serviço
            key: Chave da configuração
            value: Valor a definir
        """
        if service not in self.config:
            self.config[service] = {}
        
        self.config[service][key] = value
        self._save_config(self.config)
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        Obtém API key de um serviço (com fallback para env var)
        
        Args:
            service: Nome do serviço
            
        Returns:
            API key ou None
        """
        # Primeiro tenta do arquivo de config
        api_key = self.get(service, "api_key", "")
        
        # Se vazia, tenta variável de ambiente
        if not api_key:
            env_key = f"{service.upper()}_API_KEY".replace("-", "_")
            api_key = os.getenv(env_key, "")
        
        return api_key if api_key else None
    
    def has_api_key(self, service: str) -> bool:
        """Verifica se tem API key configurada"""
        return self.get_api_key(service) is not None
    
    def is_service_enabled(self, service: str) -> bool:
        """Verifica se serviço está habilitado e tem API key"""
        if service == "openrouter":
            return self.has_api_key("openrouter") and self.get("system", "use_ai", True)
        elif service == "yahoo_finance":
            return True  # Não precisa API key
        elif service == "coingecko":
            return True  # Funciona sem API key (mas com rate limit)
        elif service == "exchangerate":
            return True  # Funciona sem API key
        return False


# Instância global de configuração
_config_instance: Optional[APIConfig] = None


def get_api_config() -> APIConfig:
    """Obtém instância global de configuração (singleton)"""
    global _config_instance
    if _config_instance is None:
        _config_instance = APIConfig()
    return _config_instance


def reload_config():
    """Recarrega configuração"""
    global _config_instance
    _config_instance = None
    return get_api_config()

