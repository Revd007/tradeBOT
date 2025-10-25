import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class NewsAPI:
    """News API client for forex-related news with enhanced sentiment analysis"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"
        self.forex_keywords = [
            'gold', 'BTCUSDm', 'forex', 'fed', 'dollar', 'currency',
            'interest rate', 'inflation', 'central bank', 'trading'
        ]
        
        # 🔥 ENHANCED: Gold-specific keywords for better relevance
        self.gold_keywords = [
            'gold', 'precious metals', 'BTC', 'safe haven', 
            'gold price', 'gold trading', 'bullion'
        ]
    
    def get_historical_news(
        self,
        symbol: str = 'BTCUSDm',
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        language: str = 'en'
    ) -> List[Dict]:
        """
        🔥 NEW: Get historical news for training data date range
        
        Args:
            symbol: Trading symbol
            start_date: Start date for news
            end_date: End date for news
            language: Language code
        
        Returns:
            List of news articles with sentiment
        """
        if not self.api_key or self.api_key in ['free_key_or_mock', 'your_api_key_here']:
            logger.warning("⚠️  NewsAPI key not configured, using synthetic data")
            return []
        
        try:
            # Map symbols to keywords
            keyword_map = {
                'BTCUSDm': 'gold OR "gold price" OR "precious metals"',
                'EURUSD': 'euro OR "EUR/USD" OR "european central bank"',
                'GBPUSD': 'pound OR "GBP/USD" OR "bank of england"',
                'USDJPY': 'yen OR "USD/JPY" OR "bank of japan"'
            }
            
            query = keyword_map.get(symbol, 'gold')
            
            # NewsAPI free tier only allows 1 month back
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            # Limit to 1 month for free tier
            if (end_date - start_date).days > 30:
                start_date = end_date - timedelta(days=30)
                logger.warning(f"⚠️  NewsAPI free tier limited to 30 days, using {start_date.date()} to {end_date.date()}")
            
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'publishedAt',
                'language': language,
                'apiKey': self.api_key,
                'pageSize': 100  # Max results
            }
            
            response = requests.get(
                f"{self.base_url}/everything",
                params=params,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"NewsAPI HTTP {response.status_code}: {response.text[:200]}")
                return []
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            
            # Process articles with enhanced sentiment
            processed = []
            for article in articles:
                try:
                    published_at = datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00'))
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                    sentiment = self._analyze_sentiment(text)
                    
                    processed.append({
                        'datetime': published_at,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', ''),
                        'sentiment': sentiment
                    })
                except Exception as e:
                    continue
            
            logger.info(f"✅ Fetched {len(processed)} news articles from NewsAPI ({start_date.date()} to {end_date.date()})")
            return processed
        
        except Exception as e:
            logger.error(f"Error fetching historical news: {str(e)}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """
        🔥 ENHANCED: Sentiment analysis with gold-specific keywords
        """
        text_lower = text.lower()
        
        # 🔥 BULLISH KEYWORDS (expanded for gold)
        positive_words = [
            'gain', 'rise', 'surge', 'rally', 'boost', 'strong',
            'bullish', 'recovery', 'growth', 'advance', 'positive',
            'soar', 'climb', 'jump', 'outperform', 'breakout',
            'support', 'haven', 'demand', 'buying', 'uptrend',
            'inflation hedge', 'safe haven', 'record high'
        ]
        
        # 🔥 BEARISH KEYWORDS (expanded for gold)
        negative_words = [
            'fall', 'drop', 'decline', 'plunge', 'weak', 'bearish',
            'crisis', 'concern', 'fear', 'slump', 'negative', 'risk',
            'tumble', 'crash', 'slide', 'pressure', 'sell-off',
            'resistance', 'overbought', 'downtrend', 'correction',
            'dollar strength', 'rate hike'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {'label': 'NEUTRAL', 'score': 0.0, 'positive_count': 0, 'negative_count': 0}
        
        # Normalized score [-1.0, 1.0]
        score = (positive_count - negative_count) / total
        
        if score > 0.2:
            label = 'BULLISH'
        elif score < -0.2:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'
        
        return {
            'label': label,
            'score': score,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def calculate_aggregated_sentiment(
        self,
        articles: List[Dict],
        time_window_hours: int = 24
    ) -> Dict:
        """
        🔥 NEW: Calculate aggregated market sentiment from multiple articles
        
        Args:
            articles: List of news articles with sentiment
            time_window_hours: Consider articles within this time window
        
        Returns:
            Aggregated sentiment metrics
        """
        if not articles:
            return {
                'overall_label': 'NEUTRAL',
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'article_count': 0,
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0
            }
        
        # Filter by time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_articles = []
        for a in articles:
            article_time = a.get('datetime', cutoff_time)
            # 🔥 FIX: Handle timezone-aware vs naive datetime comparison
            if hasattr(article_time, 'tzinfo') and article_time.tzinfo is not None:
                article_time = article_time.replace(tzinfo=None)  # Convert to naive
            if article_time >= cutoff_time:
                recent_articles.append(a)
        
        if not recent_articles:
            recent_articles = articles[:10]  # Use 10 most recent if none in window
        
        sentiments = [a['sentiment'] for a in recent_articles]
        
        bullish_count = sum(1 for s in sentiments if s['label'] == 'BULLISH')
        bearish_count = sum(1 for s in sentiments if s['label'] == 'BEARISH')
        neutral_count = sum(1 for s in sentiments if s['label'] == 'NEUTRAL')
        
        total = len(sentiments)
        
        # Calculate ratios
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        neutral_ratio = neutral_count / total
        
        # Aggregate sentiment score (weighted average)
        sentiment_scores = [s['score'] for s in sentiments]
        avg_sentiment_score = np.mean(sentiment_scores)
        
        # Determine overall label
        if bullish_ratio > bearish_ratio * 1.5:
            overall_label = 'BULLISH'
            confidence = bullish_ratio
        elif bearish_ratio > bullish_ratio * 1.5:
            overall_label = 'BEARISH'
            confidence = bearish_ratio
        else:
            overall_label = 'NEUTRAL'
            confidence = neutral_ratio
        
        return {
            'overall_label': overall_label,
            'sentiment_score': float(avg_sentiment_score),
            'confidence': float(confidence),
            'article_count': total,
            'bullish_ratio': float(bullish_ratio),
            'bearish_ratio': float(bearish_ratio),
            'neutral_ratio': float(neutral_ratio)
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
    ) -> tuple[bool, Optional[dict]]:
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
    
    def should_trade(self, symbol: str) -> tuple[bool, str, dict]:
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
    news = news_api.get_latest_news("BTCUSDm", hours_ago=12)
    print(f"Found {len(news)} news articles")
    
    for article in news[:3]:
        print(f"\n{article['title']}")
        print(f"Sentiment: {article['sentiment']['label']} ({article['sentiment']['score']:.2f})")
    
    # Get market sentiment
    sentiment = news_api.get_market_sentiment("BTCUSDm")
    print(f"\nOverall sentiment: {sentiment}")
    
    # Test calendar
    calendar = EconomicCalendar()
    events = calendar.get_upcoming_events(hours_ahead=24)
    
    print(f"\nUpcoming events: {len(events)}")
    for event in events[:5]:
        print(f"- {event['event']} ({event['importance']}) in {event['time_until']:.0f} min")
    
    # Test news filter
    news_filter = NewsFilter("your_api_key", None)
    can_trade, reason, info = news_filter.should_trade("BTCUSDm")
    
    print(f"\nCan trade: {can_trade}")
    print(f"Reason: {reason}")
    print(f"Impact score: {info.get('impact_score', 0):.1f}/10")