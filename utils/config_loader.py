import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage application configuration"""
    
    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.load()
    
    def load(self) -> Dict:
        """Load configuration from YAML file"""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {self.config_file}, using defaults")
            self.config = self._get_default_config()
            return self.config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
            # Validate config
            if not self.validate():
                logger.error("Configuration validation failed")
                self.config = self._get_default_config()
            
            return self.config
        
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            self.config = self._get_default_config()
            return self.config
        
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = self._get_default_config()
            return self.config
    
    def validate(self) -> bool:
        """Validate configuration structure"""
        required_sections = ['trading', 'risk_management', 'strategies']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required config section: {section}")
                return False
        
        # Validate trading section
        trading = self.config.get('trading', {})
        if not trading.get('default_symbol'):
            logger.error("Missing default_symbol in trading config")
            return False
        
        # Validate risk management
        risk = self.config.get('risk_management', {})
        if not isinstance(risk.get('default_risk_percent'), (int, float)):
            logger.error("Invalid default_risk_percent in risk_management")
            return False
        
        return True
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Example: config.get('trading.default_symbol')
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        save_path = file_path or self.config_file
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'trading': {
                'default_symbol': 'XAUUSDm',
                'available_symbols': ['XAUUSDm', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD'],
                'timeframes': {
                    'fast': 'M5',
                    'medium': 'M15',
                    'slow': 'H1',
                    'trend': 'H4',
                    'daily': 'D1'
                },
                'modes': ['NORMAL', 'AGGRESSIVE', 'SCALPING', 'LONG_HOLD']
            },
            'risk_management': {
                'default_risk_percent': 1.0,
                'max_risk_percent': 3.0,
                'max_daily_loss': 100,
                'max_trades_per_day': 1000,
                'max_open_positions': 5,
                'position_sizing': {
                    'min_lot': 0.01,
                    'max_lot': 1.0,
                    'lot_step': 0.01
                },
                'stop_loss': {
                    'min_pips': 10,
                    'max_pips': 100,
                    'atr_multiplier': 2.0
                },
                'take_profit': {
                    'min_rr_ratio': 1.5,
                    'max_rr_ratio': 5.0,
                    'partial_close': True,
                    'partial_levels': [0.5, 0.3, 0.2]
                }
            },
            'strategies': {
                'counter_trend': {
                    'enabled': True,
                    'risk_level': 'extreme',
                    'min_confidence': 0.70,
                    'entry_zone_pips': 5
                },
                'breakout': {
                    'enabled': True,
                    'risk_level': 'high',
                    'min_confidence': 0.65,
                    'breakout_confirmation_bars': 2
                },
                'fibonacci_atr': {
                    'enabled': True,
                    'risk_level': 'medium',
                    'min_confidence': 0.60,
                    'fib_levels': [0.236, 0.382, 0.500, 0.618]
                }
            },
            'ai_analysis': {
                'cls_model': {
                    'enabled': True,
                    'min_confidence': 0.60,
                    'retrain_interval_days': 7
                },
                'llm': {
                    'enabled': False,
                    'model_type': 'llama',
                    'max_tokens': 512,
                    'temperature': 0.3
                },
                'trend_fusion': {
                    'enabled': True,
                    'use_meta_analysis': True,
                    'consensus_threshold': 0.65
                }
            },
            'news_filter': {
                'enabled': True,
                'pause_before_high_impact': 30,
                'resume_after_high_impact': 15,
                'impact_levels': {
                    'high': ['NFP', 'FOMC', 'CPI', 'GDP'],
                    'medium': ['Retail Sales', 'PMI', 'Unemployment'],
                    'low': ['Housing Data']
                }
            },
            'monitoring': {
                'telegram': {
                    'enabled': False,
                    'send_entry_signals': True,
                    'send_exit_signals': True,
                    'send_daily_summary': True,
                    'send_error_alerts': True
                },
                'performance_tracking': {
                    'enabled': True,
                    'save_to_firebase': False,
                    'metrics_interval': '1h'
                },
                'logging': {
                    'level': 'INFO',
                    'save_to_file': True,
                    'max_file_size_mb': 10,
                    'backup_count': 5
                }
            },
            'connection': {
                'mt5': {
                    'auto_reconnect': True,
                    'reconnect_attempts': 5,
                    'reconnect_delay': 30,
                    'ping_interval': 60
                },
                'api': {
                    'timeout': 30,
                    'retry_attempts': 3,
                    'rate_limit_delay': 1
                }
            },
            'backtesting': {
                'enabled': True,
                'default_period': 90,
                'commission_per_lot': 3.0,
                'spread_buffer_pips': 2,
                'slippage_pips': 1
            }
        }
    
    def reload(self):
        """Reload configuration from file"""
        return self.load()
    
    def __getitem__(self, key):
        """Allow dict-like access"""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Allow dict-like assignment"""
        self.config[key] = value
    
    def __contains__(self, key):
        """Allow 'in' operator"""
        return key in self.config