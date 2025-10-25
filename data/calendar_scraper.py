"""
Economic Calendar Scraper
Fetches high-impact economic events from TradingEconomics and other sources
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from bs4 import BeautifulSoup
import json

logger = logging.getLogger(__name__)


class EconomicCalendarScraper:
    """Scrapes economic calendar for high-impact events"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # 🔥 NEW: Cache to avoid repeated API calls during training
        self._calendar_cache = {}  # {date_key: events}
        self._cache_duration = 3600  # 1 hour cache
    
    def get_calendar_events(
        self, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        importance: str = 'high'
    ) -> List[Dict]:
        """
        Get economic calendar events
        
        Args:
            start_date: Start date for events
            end_date: End date for events
            importance: Filter by importance (low, medium, high)
        
        Returns:
            List of calendar events
        """
        if not start_date:
            start_date = datetime.now()
        if not end_date:
            end_date = start_date + timedelta(days=7)
        
        # 🔥 NEW: Check cache first to avoid repeated API calls
        cache_key = f"{start_date.date()}_{end_date.date()}_{importance}"
        if cache_key in self._calendar_cache:
            cached_time, cached_events = self._calendar_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_duration:
                return cached_events  # Return cached result
        
        events = []
        
        # 🔥 ONLY use Trading Economics API (disable Forex Factory)
        if self.api_key:
            events = self._get_from_trading_economics(start_date, end_date)
        else:
            logger.warning("⚠️  No TradingEconomics API key provided, returning empty calendar")
        
        # 🔥 DISABLED: Forex Factory scraping (too slow + SSL errors)
        # if not events:
        #     events = self._get_from_forex_factory(start_date, end_date)
        
        # Filter by importance
        if importance:
            events = [e for e in events if e['importance'].lower() == importance.lower()]
        
        # 🔥 NEW: Store in cache
        self._calendar_cache[cache_key] = (datetime.now(), events)
        
        return events
    
    def _get_from_trading_economics(
        self, 
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Fetch from TradingEconomics API"""
        try:
            # 🔥 FIXED API FORMAT: Use correct TradingEconomics endpoint
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # According to docs: https://docs.tradingeconomics.com/economic_calendar/snapshot/
            # Format: /calendar?c={api_key}&d1={start}&d2={end}
            endpoint = f"{self.base_url}/calendar"
            
            params = {
                'c': self.api_key,  # 🔥 FIX: Use 'c' parameter for API key
                'd1': start_str,
                'd2': end_str
            }
            
            # 🔥 DEBUG: Log the actual URL being called (once)
            if not hasattr(self, '_url_logged'):
                logger.info(f"   📡 TradingEconomics API URL: {endpoint}?c={self.api_key[:10]}...&d1={start_str}&d2={end_str}")
                self._url_logged = True
            
            # Make request
            response = self.session.get(endpoint, params=params, timeout=10)
            
            response.raise_for_status()
            
            data = response.json()
            
            # 🔥 DEBUG: Log response structure
            logger.info(f"   📊 TradingEconomics response type: {type(data)}")
            if isinstance(data, dict):
                logger.info(f"   📊 Response keys: {list(data.keys())}")
                if 'message' in data:
                    logger.warning(f"   📊 API message: {data['message']}")
            elif isinstance(data, list):
                logger.info(f"   📊 Response length: {len(data)}")
                if len(data) > 0:
                    logger.info(f"   📊 Sample item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
            
            # Check if response is valid
            if not isinstance(data, list):
                logger.warning(f"⚠️  Unexpected API response format: {type(data)}")
                return []
            
            events = []
            for item in data:
                try:
                    # 🔥 FIX: Handle different date formats from TradingEconomics
                    date_str = item.get('Date', '')
                    if not date_str:
                        continue
                    
                    # Try different date parsing methods
                    try:
                        # Format: "2024-10-13T14:30:00Z" or "2024-10-13T14:30:00"
                        if 'T' in date_str:
                            if date_str.endswith('Z'):
                                event_datetime = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            else:
                                event_datetime = datetime.fromisoformat(date_str)
                        else:
                            # Format: "2024-10-13"
                            event_datetime = datetime.strptime(date_str, '%Y-%m-%d')
                    except ValueError:
                        # Skip if date parsing fails
                        continue
                    
                    events.append({
                        'datetime': event_datetime,
                        'currency': item.get('Country', 'USD'),
                        'event': item.get('Event', ''),
                        'importance': self._map_importance(item.get('Importance', 1)),
                        'actual': item.get('Actual'),
                        'forecast': item.get('Forecast'),
                        'previous': item.get('Previous'),
                        'source': 'TradingEconomics'
                    })
                except Exception as parse_err:
                    # Skip malformed events
                    continue
            
            if events:
                logger.info(f"✅ Fetched {len(events)} events from TradingEconomics")
            else:
                logger.warning(f"⚠️  No events parsed from API response")
            
            return events
        
        except requests.exceptions.HTTPError as e:
            # 🔥 IMPROVED: Log detailed error info
            logger.warning(f"⚠️  TradingEconomics API HTTP {e.response.status_code}")
            try:
                error_msg = e.response.json().get('message', e.response.text[:200])
                logger.warning(f"   Error: {error_msg}")
            except:
                logger.warning(f"   Response: {e.response.text[:200]}")
            
            logger.warning(f"   💡 API key: {self.api_key[:10]}... (first 10 chars)")
            return []
        
        except Exception as e:
            # 🔥 IMPROVED: Silent failure for training (don't spam console)
            if not hasattr(self, '_generic_error_logged'):
                logger.warning(f"⚠️  Calendar fetch failed: {type(e).__name__}. Continuing without calendar data.")
                self._generic_error_logged = True
            return []
    
    def _get_from_forex_factory(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Fallback: Scrape from Forex Factory"""
        try:
            url = "https://www.forexfactory.com/calendar"
            
            # Forex Factory uses week-based calendar
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            events = []
            calendar_rows = soup.find_all('tr', class_='calendar__row')
            
            current_date = None
            
            for row in calendar_rows:
                # Extract date
                date_cell = row.find('td', class_='calendar__date')
                if date_cell and date_cell.text.strip():
                    date_str = date_cell.text.strip()
                    try:
                        current_date = datetime.strptime(date_str, '%b %d')
                        current_date = current_date.replace(year=datetime.now().year)
                    except:
                        pass
                
                if not current_date:
                    continue
                
                # Extract time
                time_cell = row.find('td', class_='calendar__time')
                time_str = time_cell.text.strip() if time_cell else ''
                
                # Extract currency
                currency_cell = row.find('td', class_='calendar__currency')
                currency = currency_cell.text.strip() if currency_cell else ''
                
                # Extract impact
                impact_cell = row.find('td', class_='calendar__impact')
                impact_span = impact_cell.find('span') if impact_cell else None
                importance = 'low'
                if impact_span:
                    if 'icon--ff-impact-red' in impact_span.get('class', []):
                        importance = 'high'
                    elif 'icon--ff-impact-ora' in impact_span.get('class', []):
                        importance = 'medium'
                
                # Extract event name
                event_cell = row.find('td', class_='calendar__event')
                event_name = event_cell.text.strip() if event_cell else ''
                
                # Extract actual, forecast, previous
                actual_cell = row.find('td', class_='calendar__actual')
                forecast_cell = row.find('td', class_='calendar__forecast')
                previous_cell = row.find('td', class_='calendar__previous')
                
                actual = actual_cell.text.strip() if actual_cell else None
                forecast = forecast_cell.text.strip() if forecast_cell else None
                previous = previous_cell.text.strip() if previous_cell else None
                
                # Parse datetime
                event_datetime = current_date
                if time_str and time_str != 'All Day':
                    try:
                        time_obj = datetime.strptime(time_str, '%I:%M%p')
                        event_datetime = current_date.replace(
                            hour=time_obj.hour,
                            minute=time_obj.minute
                        )
                    except:
                        pass
                
                # Filter by date range
                if start_date <= event_datetime <= end_date + timedelta(days=1):
                    events.append({
                        'datetime': event_datetime,
                        'currency': currency,
                        'event': event_name,
                        'importance': importance,
                        'actual': actual,
                        'forecast': forecast,
                        'previous': previous,
                        'source': 'ForexFactory'
                    })
            
            logger.info(f"✅ Scraped {len(events)} events from Forex Factory")
            return events
        
        except Exception as e:
            logger.error(f"Error scraping Forex Factory: {str(e)}")
            return []
    
    def _map_importance(self, value: int) -> str:
        """Map numeric importance to string"""
        if value >= 3:
            return 'high'
        elif value == 2:
            return 'medium'
        else:
            return 'low'
    
    def get_upcoming_high_impact(
        self,
        hours_ahead: int = 24,
        currencies: List[str] = None
    ) -> List[Dict]:
        """
        Get upcoming high-impact events
        
        Args:
            hours_ahead: Look ahead this many hours
            currencies: Filter by currencies (e.g., ['USD', 'EUR'])
        """
        if currencies is None:
            currencies = ['USD', 'EUR', 'GBP', 'JPY', 'AUD', 'NZD', 'CAD', 'CHF']
        
        start = datetime.now()
        end = start + timedelta(hours=hours_ahead)
        
        events = self.get_calendar_events(start, end, importance='high')
        
        # Filter by currencies
        events = [e for e in events if e['currency'] in currencies]
        
        # Sort by datetime
        events.sort(key=lambda x: x['datetime'])
        
        return events
    
    def check_news_conflict(
        self,
        symbol: str,
        buffer_minutes: int = 30
    ) -> Dict:
        """
        Check if there are high-impact news events soon
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDm, EURUSD)
            buffer_minutes: Minutes before/after event to avoid trading
        
        Returns:
            {
                'has_conflict': bool,
                'next_event': dict or None,
                'minutes_until': int,
                'should_pause': bool
            }
        """
        # Determine relevant currencies from symbol
        currencies = self._get_symbol_currencies(symbol)
        
        # Get upcoming events
        upcoming = self.get_upcoming_high_impact(hours_ahead=2, currencies=currencies)
        
        if not upcoming:
            return {
                'has_conflict': False,
                'next_event': None,
                'minutes_until': None,
                'should_pause': False
            }
        
        # Check first event
        next_event = upcoming[0]
        now = datetime.now()
        minutes_until = (next_event['datetime'] - now).total_seconds() / 60
        
        # Should pause if within buffer time
        should_pause = 0 <= minutes_until <= buffer_minutes
        
        return {
            'has_conflict': should_pause,
            'next_event': next_event,
            'minutes_until': int(minutes_until),
            'should_pause': should_pause
        }
    
    def _get_symbol_currencies(self, symbol: str) -> List[str]:
        """Extract currencies from trading symbol"""
        # Gold/Silver symbols
        if symbol.startswith('BTC') or symbol.startswith('XAG'):
            return ['USD']
        
        # Forex pairs
        if len(symbol) == 6:
            return [symbol[:3], symbol[3:]]
        
        # Default to major currencies
        return ['USD', 'EUR', 'GBP']
    
    def get_daily_summary(self) -> str:
        """Get human-readable summary of today's high-impact events"""
        start = datetime.now().replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)
        
        events = self.get_calendar_events(start, end, importance='high')
        
        if not events:
            return "No high-impact events scheduled for today."
        
        summary = f"📅 Today's High-Impact Events ({len(events)}):\n\n"
        
        for event in events:
            time_str = event['datetime'].strftime('%H:%M')
            summary += f"• {time_str} - {event['currency']}: {event['event']}\n"
            if event.get('forecast'):
                summary += f"  Forecast: {event['forecast']}, Previous: {event.get('previous', 'N/A')}\n"
        
        return summary


class NewsImpactScorer:
    """Calculate impact score for news events"""
    
    # High impact keywords
    HIGH_IMPACT_KEYWORDS = [
        'NFP', 'Non-Farm', 'FOMC', 'Interest Rate', 'GDP',
        'CPI', 'Inflation', 'Employment', 'Unemployment',
        'Central Bank', 'Fed', 'ECB', 'BOE', 'BOJ'
    ]
    
    @staticmethod
    def calculate_impact_score(event: Dict) -> float:
        """
        Calculate news impact score (0-10)
        
        Factors:
        - Event importance
        - Keyword matching
        - Forecast vs Previous deviation
        """
        score = 0.0
        
        # Base score from importance
        importance_scores = {'low': 2, 'medium': 5, 'high': 8}
        score += importance_scores.get(event['importance'], 0)
        
        # Check for high-impact keywords
        event_name = event.get('event', '').upper()
        for keyword in NewsImpactScorer.HIGH_IMPACT_KEYWORDS:
            if keyword.upper() in event_name:
                score += 1
                break
        
        # Check forecast deviation
        try:
            forecast = float(event.get('forecast', 0))
            previous = float(event.get('previous', 0))
            
            if previous != 0:
                deviation = abs((forecast - previous) / previous) * 100
                if deviation > 10:  # > 10% deviation
                    score += 1
        except:
            pass
        
        return min(score, 10.0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test calendar scraper
    scraper = EconomicCalendarScraper()
    
    # Get upcoming high-impact events
    events = scraper.get_upcoming_high_impact(hours_ahead=48)
    
    print(f"\n{'='*60}")
    print(f"Upcoming High-Impact Events ({len(events)})")
    print(f"{'='*60}\n")
    
    for event in events[:5]:
        print(f"{event['datetime'].strftime('%Y-%m-%d %H:%M')} - {event['currency']}")
        print(f"  {event['event']}")
        print(f"  Forecast: {event.get('forecast', 'N/A')}, Previous: {event.get('previous', 'N/A')}")
        
        # Calculate impact score
        scorer = NewsImpactScorer()
        impact = scorer.calculate_impact_score(event)
        print(f"  Impact Score: {impact:.1f}/10")
        print()
    
    # Check news conflict for BTCUSDm
    conflict = scraper.check_news_conflict('BTCUSDm', buffer_minutes=30)
    
    print(f"\n{'='*60}")
    print(f"News Conflict Check for BTCUSDm")
    print(f"{'='*60}\n")
    print(f"Has Conflict: {conflict['has_conflict']}")
    print(f"Should Pause: {conflict['should_pause']}")
    if conflict['next_event']:
        print(f"Next Event: {conflict['next_event']['event']}")
        print(f"Minutes Until: {conflict['minutes_until']}")
    
    # Daily summary
    print(f"\n{scraper.get_daily_summary()}")

