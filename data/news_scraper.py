mport requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class NewsAPI:
    """News API client for forex-related news"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.forex_keywords = [
            'gold', 'xauusd', 'forex', 'fed', 'dollar', 'currency',
            'interest rate', 'inflation', 'central bank', 'trading'
        ]
    
    def get_latest_news(
        self,
        symbol: str = 'gold',
        hours_ago: int = 24,
        language: str = 'en'
    ) -> List[Dict]:
        """
        Get latest news for a symbol
        
        Returns:
            List of news articles with sentiment
        """
        try:
            # Map symbols to keywords
            keyword_map = {
                'XAUUSD': 'gold OR "gold price"',
                'EURUSD': 'euro OR "EUR/USD"',
                'GBPUSD': 'pound OR "GBP/USD"',
                'USDJPY': 'yen OR "USD/JPY"'
            }
            
            query = keyword_map.get(symbol, 'forex')
            
            # Calculate time range
            from_time = datetime.now() - timedelta(hours=hours_ago)
            
            params = {
                'q': query,
                'from': from_time.isoformat(),
                'sortBy': 'publishedAt',
                'language': language,
                'apiKey': self.api_key
            }
            
            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"News API error: {response.status_code}")
                return []
            
            data = response.json()
            articles = data.get('articles', [])
            
            # Process articles
            processed = []
            for article in articles[:10]:  # Limit to 10 most recent
                processed.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', ''),
                    'sentiment': self._analyze_sentiment(
                        article.get('title', '') + ' ' + article.get('description', '')
                    )
                })
            
            logger.info(f"Retrieved {len(processed)} news articles for {symbol}")
            return processed
        
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        Simple sentiment analysis (keyword-based)
        In production, use NLP library like TextBlob or transformers
        """
        text_lower = text.lower()
        
        positive_words = [
            'gain', 'rise', 'surge', 'rally', 'boost', 'strong',
            'bullish', 'recovery', 'growth', 'advance', 'positive'
        ]
        
        negative_words = [
            'fall', 'drop', 'decline', 'plunge', 'weak', 'bearish',
            'crisis', 'concern', 'fear', 'slump', 'negative', 'risk'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {'label': 'NEUTRAL', 'score': 0.0}
        
        score = (positive_count - negative_count) / total
        
        if score > 0.2:
            label = 'POSITIVE'
        elif score < -0.2:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return {
            'label': label,
            'score': score,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def get_market_sentiment(self, symbol: str, hours_ago: int = 6) -> Dict:
        """
        Calculate overall market sentiment
        
        Returns:
            {
                'sentiment': 'BULLISH', 'BEARISH', or 'NEUTRAL',
                'confidence': 0.0 - 1.0,
                'news_count': int
            }
        """
        articles = self.get_latest_news(symbol, hours_ago)
        
        if not articles:
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'news_count': 0
            }
        
        sentiments = [a['sentiment'] for a in articles]
        
        positive_count = sum(1 for s in sentiments if s['label'] == 'POSITIVE')
        negative_count = sum(1 for s in sentiments if s['label'] == 'NEGATIVE')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'NEUTRAL')
        
        total = len(sentiments)
        
        if positive_count > negative_count * 1.5:
            sentiment = 'BULLISH'
            confidence = positive_count / total
        elif negative_count > positive_count * 1.5:
            sentiment = 'BEARISH'
            confidence = negative_count / total
        else:
            sentiment = 'NEUTRAL'
            confidence = neutral_count / total if neutral_count > 0 else 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'news_count': total,
            'breakdown': {
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count
            }
        }


# data/calendar_scraper.py
"""
Economic Calendar Integration (TradingEconomics)
"""

class EconomicCalendar:
    """Economic calendar for high-impact events"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.tradingeconomics.com"
        
        # High impact events
        self.high_impact_events = [
            'Non-Farm Payrolls', 'NFP', 'FOMC', 'Interest Rate Decision',
            'CPI', 'Consumer Price Index', 'GDP', 'Unemployment Rate',
            'Retail Sales', 'FOMC Minutes', 'ECB Press Conference'
        ]
    
    def get_upcoming_events(
        self,
        country: str = 'United States',
        hours_ahead: int = 24
    ) -> List[Dict]:
        """Get upcoming economic events"""
        try:
            if not self.api_key:
                # Return mock data if no API key
                return self._get_mock_events()
            
            end_time = datetime.now() + timedelta(hours=hours_ahead)
            
            params = {
                'c': country,
                'f': 'json',
                'key': self.api_key
            }
            
            response = requests.get(
                f"{self.base_url}/calendar",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Calendar API error: {response.status_code}")
                return []
            
            events = response.json()
            
            # Filter upcoming events
            upcoming = []
            for event in events:
                event_time = datetime.fromisoformat(event.get('Date', ''))
                
                if event_time > datetime.now() and event_time < end_time:
                    upcoming.append({
                        'event': event.get('Event', ''),
                        'country': event.get('Country', ''),
                        'date': event.get('Date', ''),
                        'actual': event.get('Actual'),
                        'forecast': event.get('Forecast'),
                        'previous': event.get('Previous'),
                        'importance': self._get_importance(event.get('Event', '')),
                        'time_until': (event_time - datetime.now()).total_seconds() / 60
                    })
            
            return sorted(upcoming, key=lambda x: x['time_until'])
        
        except Exception as e:
            logger.error(f"Error fetching calendar: {str(e)}")
            return []
    
    def _get_importance(self, event_name: str) -> str:
        """Determine event importance"""
        for high_event in self.high_impact_events:
            if high_event.lower() in event_name.lower():
                return 'HIGH'
        
        medium_events = ['PMI', 'Manufacturing', 'Services', 'Trade Balance']
        for med_event in medium_events:
            if med_event.lower() in event_name.lower():
                return 'MEDIUM'
        
        return 'LOW'
    
    def should_pause_trading(
        self,
        symbol: str,
        minutes_before: int = 30,
        minutes_after: int = 15
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Check if trading should be paused due to upcoming news
        
        Returns:
            (should_pause: bool, event: Dict or None)
        """
        events = self.get_upcoming_events(hours_ahead=2)
        
        for event in events:
            if event['importance'] == 'HIGH':
                time_until = event['time_until']
                
                # Check if within pause window
                if -minutes_after <= time_until <= minutes_before:
                    logger.warning(f"⚠️ High impact event in {time_until:.0f} minutes: {event['event']}")
                    return True, event
        
        return False, None
    
    def get_news_impact_score(self, hours_ahead: int = 4) -> float:
        """
        Calculate news impact score (0-10)
        Higher score = more risk
        """
        events = self.get_upcoming_events(hours_ahead=hours_ahead)
        
        impact_score = 0.0
        
        for event in events:
            time_until = event['time_until']
            importance = event['importance']
            
            # Score based on importance and proximity
            if importance == 'HIGH':
                base_score = 5.0
            elif importance == 'MEDIUM':
                base_score = 2.0
            else:
                base_score = 0.5
            
            # Increase score as event gets closer
            time_factor = max(0, 1 - (time_until / 240))  # 4 hours window
            impact_score += base_score * time_factor
        
        return min(10.0, impact_score)
    
    def _get_mock_events(self) -> List[Dict]:
        """Return mock events for testing"""
        return [
            {
                'event': 'CPI (Consumer Price Index)',
                'country': 'United States',
                'date': (datetime.now() + timedelta(hours=5)).isoformat(),
                'importance': 'HIGH',
                'time_until': 300
            },
            {
                'event': 'Retail Sales',
                'country': 'United States',
                'date': (datetime.now() + timedelta(hours=10)).isoformat(),
                'importance': 'MEDIUM',
                'time_until': 600
            }
        ]


# data/news_filter.py
"""
News Filter - Combines news and calendar data
"""

class NewsFilter:
    """Filter trades based on news and economic events"""
    
    def __init__(self, news_api_key: Optional[str], calendar_api_key: Optional[str] = None):
        self.news_api = NewsAPI(news_api_key) if news_api_key else None
        self.calendar = EconomicCalendar(calendar_api_key)
    
    def should_trade(self, symbol: str) -> Tuple[bool, str, Dict]:
        """
        Determine if it's safe to trade based on news
        
        Returns:
            (can_trade: bool, reason: str, info: Dict)
        """
        info = {
            'news_sentiment': None,
            'upcoming_events': [],
            'impact_score': 0.0
        }
        
        # Check upcoming high-impact events
        should_pause, event = self.calendar.should_pause_trading(symbol)
        
        if should_pause:
            return False, f"High impact event: {event['event']}", info
        
        # Get news impact score
        impact_score = self.calendar.get_news_impact_score()
        info['impact_score'] = impact_score
        
        if impact_score > 7.0:
            return False, f"High news impact score: {impact_score:.1f}/10", info
        
        # Get news sentiment
        if self.news_api:
            sentiment = self.news_api.get_market_sentiment(symbol)
            info['news_sentiment'] = sentiment
        
        # Get upcoming events
        info['upcoming_events'] = self.calendar.get_upcoming_events(hours_ahead=24)
        
        return True, "OK to trade", info
    
    def get_trading_context(self, symbol: str) -> Dict:
        """Get complete trading context"""
        can_trade, reason, info = self.should_trade(symbol)
        
        return {
            'can_trade': can_trade,
            'reason': reason,
            'news_sentiment': info.get('news_sentiment'),
            'impact_score': info.get('impact_score', 0.0),
            'upcoming_events': info.get('upcoming_events', [])[:5],  # Top 5 events
            'risk_level': 'HIGH' if info.get('impact_score', 0) > 5 else 'NORMAL'
        }


if __name__ == "__main__":
    # Test news and calendar
    logging.basicConfig(level=logging.INFO)
    
    # Mock API key for testing
    news_api = NewsAPI("your_api_key_here")
    
    # Get latest news
    news = news_api.get_latest_news("XAUUSD", hours_ago=12)
    print(f"Found {len(news)} news articles")
    
    for article in news[:3]:
        print(f"\n{article['title']}")
        print(f"Sentiment: {article['sentiment']['label']} ({article['sentiment']['score']:.2f})")
    
    # Get market sentiment
    sentiment = news_api.get_market_sentiment("XAUUSD")
    print(f"\nOverall sentiment: {sentiment}")
    
    # Test calendar
    calendar = EconomicCalendar()
    events = calendar.get_upcoming_events(hours_ahead=24)
    
    print(f"\nUpcoming events: {len(events)}")
    for event in events[:5]:
        print(f"- {event['event']} ({event['importance']}) in {event['time_until']:.0f} min")
    
    # Test news filter
    news_filter = NewsFilter("your_api_key", None)
    can_trade, reason, info = news_filter.should_trade("XAUUSD")
    
    print(f"\nCan trade: {can_trade}")
    print(f"Reason: {reason}")
    print(f"Impact score: {info.get('impact_score', 0):.1f}/10")