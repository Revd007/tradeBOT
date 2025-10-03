"""
Connection Manager for MT5
Manages MT5 connection health with auto-reconnection and monitoring
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Callable, Optional
from core.mt5_handler import MT5Handler

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages MT5 connection with health checks and auto-reconnect"""
    
    def __init__(
        self, 
        mt5_handler: MT5Handler, 
        ping_interval: int = 60,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 5,
        reconnect_delay: int = 10
    ):
        """
        Args:
            mt5_handler: MT5Handler instance
            ping_interval: Seconds between health checks
            auto_reconnect: Enable automatic reconnection
            max_reconnect_attempts: Maximum reconnection attempts before giving up
            reconnect_delay: Seconds to wait between reconnection attempts
        """
        self.mt5 = mt5_handler
        self.ping_interval = ping_interval
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        
        self.last_health_check = None
        self.reconnect_attempts = 0
        self.is_monitoring = False
        self.monitor_thread = None
        
        self.connection_lost_callbacks = []
        self.connection_restored_callbacks = []
        
        self.stats = {
            'total_disconnects': 0,
            'total_reconnects': 0,
            'last_disconnect': None,
            'last_reconnect': None,
            'uptime_start': datetime.now()
        }
    
    def health_check(self) -> bool:
        """
        Perform health check on MT5 connection
        
        Returns:
            True if connection is healthy, False otherwise
        """
        current_time = datetime.now()
        
        # Skip if checked recently
        if (self.last_health_check and 
            (current_time - self.last_health_check).seconds < self.ping_interval):
            return True
        
        self.last_health_check = current_time
        
        # Ping MT5
        try:
            if not self.mt5.ping():
                logger.warning("⚠️ MT5 ping failed")
                self._handle_connection_lost()
                
                if self.auto_reconnect:
                    return self.attempt_reconnect()
                return False
            
            # Reset reconnect counter on successful ping
            if self.reconnect_attempts > 0:
                self.reconnect_attempts = 0
                logger.info("✅ Connection stable")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Health check error: {str(e)}")
            return False
    
    def attempt_reconnect(self) -> bool:
        """
        Attempt to reconnect to MT5
        
        Returns:
            True if reconnection successful, False otherwise
        """
        self.reconnect_attempts += 1
        
        if self.reconnect_attempts > self.max_reconnect_attempts:
            logger.error(
                f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) reached. "
                "Giving up."
            )
            return False
        
        logger.info(
            f"🔄 Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}..."
        )
        
        # Wait before attempting
        time.sleep(self.reconnect_delay)
        
        # Attempt reconnection
        if self.mt5.reconnect():
            logger.info("✅ Reconnection successful!")
            self._handle_connection_restored()
            self.reconnect_attempts = 0
            self.stats['total_reconnects'] += 1
            self.stats['last_reconnect'] = datetime.now()
            return True
        else:
            logger.error(f"❌ Reconnection attempt {self.reconnect_attempts} failed")
            return False
    
    def start_monitoring(self):
        """Start background monitoring thread"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"✅ Started connection monitoring (interval: {self.ping_interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("⏹️ Stopped connection monitoring")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                self.health_check()
                time.sleep(self.ping_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(5)
    
    def _handle_connection_lost(self):
        """Handle connection lost event"""
        self.stats['total_disconnects'] += 1
        self.stats['last_disconnect'] = datetime.now()
        
        logger.warning("❌ MT5 connection lost!")
        
        # Execute callbacks
        for callback in self.connection_lost_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in connection_lost callback: {str(e)}")
    
    def _handle_connection_restored(self):
        """Handle connection restored event"""
        logger.info("✅ MT5 connection restored!")
        
        # Execute callbacks
        for callback in self.connection_restored_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in connection_restored callback: {str(e)}")
    
    def on_connection_lost(self, callback: Callable):
        """Register callback for connection lost event"""
        self.connection_lost_callbacks.append(callback)
    
    def on_connection_restored(self, callback: Callable):
        """Register callback for connection restored event"""
        self.connection_restored_callbacks.append(callback)
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        uptime = datetime.now() - self.stats['uptime_start']
        
        return {
            'is_connected': self.mt5.connected,
            'total_disconnects': self.stats['total_disconnects'],
            'total_reconnects': self.stats['total_reconnects'],
            'last_disconnect': self.stats['last_disconnect'],
            'last_reconnect': self.stats['last_reconnect'],
            'uptime_seconds': int(uptime.total_seconds()),
            'uptime_formatted': str(uptime).split('.')[0],
            'is_monitoring': self.is_monitoring,
            'reconnect_attempts': self.reconnect_attempts
        }
    
    def force_reconnect(self) -> bool:
        """Force immediate reconnection"""
        logger.info("🔄 Forcing reconnection...")
        return self.mt5.reconnect()


class ConnectionHealthMonitor:
    """Advanced connection health monitoring with alerts"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.conn_mgr = connection_manager
        self.alert_callbacks = []
        
        self.thresholds = {
            'max_reconnects_per_hour': 5,
            'max_disconnect_duration': 300  # 5 minutes
        }
        
        self.recent_disconnects = []
    
    def check_health_metrics(self) -> dict:
        """
        Check advanced health metrics
        
        Returns:
            {
                'status': 'HEALTHY', 'WARNING', or 'CRITICAL',
                'issues': list of issues,
                'recommendations': list of recommendations
            }
        """
        stats = self.conn_mgr.get_connection_stats()
        issues = []
        recommendations = []
        
        # Check frequent disconnects
        if stats['total_disconnects'] > 0:
            # Count recent disconnects (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent = [d for d in self.recent_disconnects if d > one_hour_ago]
            
            if len(recent) >= self.thresholds['max_reconnects_per_hour']:
                issues.append(f"Frequent disconnects: {len(recent)} in last hour")
                recommendations.append("Check internet connection stability")
                recommendations.append("Verify MT5 server status")
        
        # Check if currently disconnected
        if not stats['is_connected']:
            issues.append("Currently disconnected from MT5")
            recommendations.append("Check if MT5 terminal is running")
            recommendations.append("Verify login credentials")
        
        # Check disconnect duration
        if stats['last_disconnect']:
            disconnect_duration = (datetime.now() - stats['last_disconnect']).seconds
            if disconnect_duration > self.thresholds['max_disconnect_duration']:
                issues.append(f"Long disconnect duration: {disconnect_duration}s")
                recommendations.append("Consider manual intervention")
        
        # Determine overall status
        if not stats['is_connected']:
            status = 'CRITICAL'
        elif len(issues) > 0:
            status = 'WARNING'
        else:
            status = 'HEALTHY'
        
        return {
            'status': status,
            'issues': issues,
            'recommendations': recommendations,
            'stats': stats
        }
    
    def alert(self, message: str, severity: str = 'INFO'):
        """Send alert through registered callbacks"""
        alert_data = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': severity
        }
        
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")
    
    def on_alert(self, callback: Callable):
        """Register alert callback"""
        self.alert_callbacks.append(callback)


if __name__ == "__main__":
    # Test connection manager
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from core.mt5_handler import MT5Handler
    
    # Initialize MT5
    mt5 = MT5Handler(
        login=12345678,
        password="your_password",
        server="MetaQuotes-Demo"
    )
    
    if mt5.initialize():
        # Create connection manager
        conn_mgr = ConnectionManager(
            mt5_handler=mt5,
            ping_interval=10,  # Check every 10 seconds
            auto_reconnect=True,
            max_reconnect_attempts=3
        )
        
        # Register callbacks
        def on_lost():
            print("\n🔴 CONNECTION LOST CALLBACK TRIGGERED")
        
        def on_restored():
            print("\n🟢 CONNECTION RESTORED CALLBACK TRIGGERED")
        
        conn_mgr.on_connection_lost(on_lost)
        conn_mgr.on_connection_restored(on_restored)
        
        # Create health monitor
        health_monitor = ConnectionHealthMonitor(conn_mgr)
        
        def on_alert(alert_data):
            print(f"\n🚨 ALERT [{alert_data['severity']}]: {alert_data['message']}")
        
        health_monitor.on_alert(on_alert)
        
        # Start monitoring
        conn_mgr.start_monitoring()
        
        print("\n" + "="*60)
        print("CONNECTION MANAGER TEST")
        print("="*60)
        print("\nMonitoring connection for 30 seconds...")
        print("Try closing MT5 terminal to test reconnection\n")
        
        try:
            for i in range(6):
                time.sleep(5)
                
                # Check health
                health = health_monitor.check_health_metrics()
                
                print(f"\n[{i*5}s] Status: {health['status']}")
                if health['issues']:
                    print(f"Issues: {', '.join(health['issues'])}")
                
                # Get stats
                stats = conn_mgr.get_connection_stats()
                print(f"Connected: {stats['is_connected']}")
                print(f"Uptime: {stats['uptime_formatted']}")
                print(f"Disconnects: {stats['total_disconnects']}")
                print(f"Reconnects: {stats['total_reconnects']}")
        
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
        
        finally:
            conn_mgr.stop_monitoring()
            mt5.shutdown()
            
            print("\n" + "="*60)
            print("FINAL STATISTICS")
            print("="*60)
            final_stats = conn_mgr.get_connection_stats()
            for key, value in final_stats.items():
                print(f"{key}: {value}")
    
    else:
        print("❌ Failed to initialize MT5")

