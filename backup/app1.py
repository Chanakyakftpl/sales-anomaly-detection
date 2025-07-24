"""
Sales Anomaly Detection API - Fixed Version Based on Original Structure
"""

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import urllib.parse

# Import configuration
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load configuration
app.config.from_object(Config)

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

@dataclass
class AnomalyData:
    date: str
    kitchen: str
    brand: str
    day_type: str
    anomaly_type: str
    expected: str
    actual: str
    deviation: str
    severity: str
    impact_score: int
    account_manager: str

class DatabaseConnection:
    def __init__(self):
        self.config = Config()
        self.engine = None
        self.connection = None
    
    def connect(self):
        try:
            # URL encode the password to handle special characters
            password_encoded = urllib.parse.quote_plus(self.config.DB_PASSWORD)
            
            # Create SQLAlchemy connection string
            connection_string = (
                f"mysql+pymysql://{self.config.DB_USER}:{password_encoded}"
                f"@{self.config.DB_HOST}/{self.config.DB_NAME}"
                f"?connect_timeout={self.config.DB_TIMEOUT}"
            )
            
            # Create SQLAlchemy engine
            self.engine = create_engine(
                connection_string,
                pool_recycle=3600,  # Recycle connections every hour
                pool_pre_ping=True,  # Verify connections before use
                echo=False  # Set to True for SQL debugging
            )
            
            # Test the connection
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()
            
            logger.info("✅ Database connection established to Azure MySQL via SQLAlchemy")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected database error: {e}")
            return False
    
    def disconnect(self):
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection pool disposed")
    
    def execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        try:
            if not self.engine:
                if not self.connect():
                    return pd.DataFrame()
            
            # Use pandas with SQLAlchemy engine
            if params:
                df = pd.read_sql(
                    sql=text(query),
                    con=self.engine,
                    params=params
                )
            else:
                df = pd.read_sql(
                    sql=text(query),
                    con=self.engine
                )
            
            logger.debug(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"SQL execution failed: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return pd.DataFrame()

class AnomalyDetector:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        self.config = Config()
        
    def get_yesterday_data(self) -> pd.DataFrame:
        """Get yesterday's order data at kitchen+brand+source level"""
        query = """
        SELECT 
            DATE(o.date) as order_date,
            o.kitchen,
            o.brand,
            o.source,
            COALESCE(am_data.AM, 'Unassigned') as account_manager,
            COUNT(*) as order_count,
            SUM(o.order_total) as total_revenue,
            AVG(o.order_total) as avg_order_value,
            SUM(o.gmv) as total_gmv,
            (SUM(o.discount) / SUM(o.gmv)) * 100 as discount_rate,
            SUM(o.discount) as total_discount,
            DAYNAME(o.date) as day_name
        FROM vw_up_only_orders o
        LEFT JOIN (
            SELECT DISTINCT kitchen, AM 
            FROM up_location_master 
            WHERE AM IS NOT NULL AND AM != ''
        ) am_data ON TRIM(LOWER(o.kitchen)) = TRIM(LOWER(am_data.kitchen))
        WHERE DATE(o.date) = :yesterday_date
        AND o.state = 'Completed'
        AND o.source IN ('swiggy', 'zomato')
        GROUP BY o.kitchen, o.brand, o.source, DATE(o.date), DAYNAME(o.date)
        ORDER BY o.kitchen, o.brand, o.source
        """
        return self.db.execute_query(query, {'yesterday_date': self.yesterday})
    
    def get_historical_data(self, days_back: int = 14) -> pd.DataFrame:
        """Get historical data at kitchen+brand+source level"""
        end_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = """
        SELECT 
            DATE(o.date) as order_date,
            o.kitchen,
            o.brand,
            o.source,
            COALESCE(am_data.AM, 'Unassigned') as account_manager,
            COUNT(*) as order_count,
            SUM(o.order_total) as total_revenue,
            AVG(o.order_total) as avg_order_value,
            SUM(o.gmv) as total_gmv,
            (SUM(o.discount) / SUM(o.gmv)) * 100 as discount_rate,
            SUM(o.discount) as total_discount,
            DAYNAME(o.date) as day_name
        FROM vw_up_only_orders o
        LEFT JOIN (
            SELECT DISTINCT kitchen, AM 
            FROM up_location_master 
            WHERE AM IS NOT NULL AND AM != ''
        ) am_data ON TRIM(LOWER(o.kitchen)) = TRIM(LOWER(am_data.kitchen))
        WHERE DATE(o.date) BETWEEN :start_date AND :end_date
        AND o.state = 'Completed'
        AND o.source IN ('swiggy', 'zomato')
        GROUP BY o.kitchen, o.brand, o.source, DATE(o.date), DAYNAME(o.date)
        ORDER BY order_date DESC, o.kitchen, o.brand, o.source
        """
        return self.db.execute_query(query, {
            'start_date': start_date,
            'end_date': end_date
        })
    
    def calculate_kitchen_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep data at kitchen+brand+source level - DO NOT AGGREGATE SOURCES"""
        if df.empty:
            return pd.DataFrame()
        
        # Return data as-is since it's already grouped by kitchen+brand+source in the query
        # DO NOT combine sources - each kitchen+brand+source should be separate
        return df
    
    def calculate_baseline_metrics(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate baseline metrics for each kitchen+brand+source combination"""
        if historical_data.empty:
            return pd.DataFrame()
        
        # Group by kitchen, brand, source, and day of week for baselines
        baseline = historical_data.groupby(['kitchen', 'brand', 'source', 'day_name']).agg({
            'order_count': ['mean', 'std'],
            'total_revenue': ['mean', 'std'],
            'avg_order_value': ['mean', 'std'],
            'total_gmv': ['mean', 'std'],
            'discount_rate': ['mean', 'std'],
            'total_discount': ['mean', 'std'],
            'account_manager': 'first'
        }).reset_index()
        
        # Flatten column names
        baseline.columns = ['kitchen', 'brand', 'source', 'day_name',
                           'avg_order_count', 'std_order_count',
                           'avg_revenue', 'std_revenue',
                           'avg_aov', 'std_aov',
                           'avg_gmv', 'std_gmv',
                           'avg_discount_rate', 'std_discount_rate',
                           'avg_discount', 'std_discount',
                           'account_manager']
        
        return baseline
    
    def detect_revenue_anomalies(self, yesterday_data: pd.DataFrame, baseline: pd.DataFrame) -> List[AnomalyData]:
        """Detect revenue spikes and drops at kitchen+brand+source level using order_total"""
        anomalies = []
        
        if yesterday_data.empty or baseline.empty:
            return anomalies
        
        for _, row in yesterday_data.iterrows():
            baseline_row = baseline[
                (baseline['kitchen'] == row['kitchen']) & 
                (baseline['brand'] == row['brand']) & 
                (baseline['source'] == row['source']) & 
                (baseline['day_name'] == row['day_name'])
            ]
            
            if baseline_row.empty:
                continue
                
            baseline_row = baseline_row.iloc[0]
            
            # Revenue deviation calculation using order_total
            expected_revenue = baseline_row['avg_revenue']
            actual_revenue = row['total_revenue']
            
            if expected_revenue > 0:
                deviation_pct = ((actual_revenue - expected_revenue) / expected_revenue) * 100
                
                # Revenue Spike Detection
                if deviation_pct > self.config.REVENUE_SPIKE_THRESHOLD:
                    severity = 'high' if deviation_pct > 120 else 'medium'
                    anomalies.append(AnomalyData(
                        date=row['order_date'].strftime('%Y-%m-%d'),
                        kitchen=row['kitchen'],  # Just kitchen name
                        brand=row['brand'],
                        day_type=row['source'],  # Use day_type field for source
                        anomaly_type='Revenue Spike',
                        expected=f"₹{expected_revenue:.0f} ({baseline_row['avg_order_count']:.0f} orders)",
                        actual=f"₹{actual_revenue:.0f} ({row['order_count']} orders)",
                        deviation=f"+{deviation_pct:.1f}%",
                        severity=severity,
                        impact_score=85 if severity == 'high' else 65,
                        account_manager=row['account_manager'] if row['account_manager'] else 'Unassigned'
                    ))
                
                # Revenue Drop Detection
                elif deviation_pct < self.config.REVENUE_DROP_THRESHOLD:
                    severity = 'high' if deviation_pct < -60 else 'medium'
                    anomalies.append(AnomalyData(
                        date=row['order_date'].strftime('%Y-%m-%d'),
                        kitchen=row['kitchen'],  # Just kitchen name
                        brand=row['brand'],
                        day_type=row['source'],  # Use day_type field for source
                        anomaly_type='Revenue Drop',
                        expected=f"₹{expected_revenue:.0f} (last week avg)",
                        actual=f"₹{actual_revenue:.0f} (yesterday)",
                        deviation=f"{deviation_pct:.1f}%",
                        severity=severity,
                        impact_score=85 if severity == 'high' else 65,
                        account_manager=row['account_manager'] or 'Unassigned'
                    ))
        
        return anomalies
    
    def detect_volume_anomalies(self, yesterday_data: pd.DataFrame, baseline: pd.DataFrame) -> List[AnomalyData]:
        """Detect order volume spikes at kitchen+brand+source level"""
        anomalies = []
        
        if yesterday_data.empty or baseline.empty:
            return anomalies
        
        for _, row in yesterday_data.iterrows():
            baseline_row = baseline[
                (baseline['kitchen'] == row['kitchen']) & 
                (baseline['brand'] == row['brand']) & 
                (baseline['source'] == row['source']) & 
                (baseline['day_name'] == row['day_name'])
            ]
            
            if baseline_row.empty:
                continue
                
            baseline_row = baseline_row.iloc[0]
            
            expected_orders = baseline_row['avg_order_count']
            actual_orders = row['order_count']
            
            if expected_orders > 0:
                deviation_pct = ((actual_orders - expected_orders) / expected_orders) * 100
                
                # Volume Spike Detection
                if deviation_pct > self.config.VOLUME_SPIKE_THRESHOLD:
                    severity = 'high' if deviation_pct > 150 else 'medium'
                    avg_aov = row['total_revenue'] / row['order_count'] if row['order_count'] > 0 else 0
                    expected_aov = baseline_row['avg_aov']
                    
                    anomalies.append(AnomalyData(
                        date=row['order_date'].strftime('%Y-%m-%d'),
                        kitchen=row['kitchen'],  # Just kitchen name
                        brand=row['brand'],
                        day_type=row['source'],  # Use day_type field for source
                        anomaly_type='Order Volume Spike',
                        expected=f"{expected_orders:.0f} orders (₹{expected_aov:.0f} avg)",
                        actual=f"{actual_orders} orders (₹{avg_aov:.0f} avg)",
                        deviation=f"+{deviation_pct:.1f}%",
                        severity=severity,
                        impact_score=85 if severity == 'high' else 65,
                        account_manager=row['account_manager'] or 'Unassigned'
                    ))
        
        return anomalies
    
    def detect_discount_anomalies(self, yesterday_data: pd.DataFrame, baseline: pd.DataFrame) -> List[AnomalyData]:
        """Detect high/low discounting anomalies using discount rate (discount/GMV)"""
        anomalies = []
        
        if yesterday_data.empty or baseline.empty:
            return anomalies
        
        for _, row in yesterday_data.iterrows():
            baseline_row = baseline[
                (baseline['kitchen'] == row['kitchen']) & 
                (baseline['brand'] == row['brand']) & 
                (baseline['source'] == row['source']) & 
                (baseline['day_name'] == row['day_name'])
            ]
            
            if baseline_row.empty or row['total_gmv'] == 0:
                continue
                
            baseline_row = baseline_row.iloc[0]
            
            # Use discount rate from query (already calculated as discount/GMV * 100)
            actual_discount_rate = row['discount_rate']
            expected_discount_rate = baseline_row['avg_discount_rate']
            
            # High Discounting Detection
            if expected_discount_rate > 0 and actual_discount_rate > expected_discount_rate * self.config.HIGH_DISCOUNT_MULTIPLIER:
                severity = 'high' if actual_discount_rate > 50 else 'medium'
                anomalies.append(AnomalyData(
                    date=row['order_date'].strftime('%Y-%m-%d'),
                    kitchen=row['kitchen'],  # Just kitchen name
                    brand=row['brand'],
                    day_type=row['source'],  # Use day_type field for source
                    anomaly_type='High Discounting',
                    expected=f"{expected_discount_rate:.1f}% discount rate",
                    actual=f"{actual_discount_rate:.1f}% discount rate",
                    deviation=f"+{((actual_discount_rate/expected_discount_rate)-1)*100:.0f}%",
                    severity=severity,
                    impact_score=85 if severity == 'high' else 65,
                    account_manager=row['account_manager'] or 'Unassigned'
                ))
        
        return anomalies
    
    def get_all_anomalies(self) -> List[AnomalyData]:
        """Main method to detect all types of anomalies - EXACTLY as original"""
        logger.info(f"🔍 Detecting anomalies for {self.yesterday}")
        
        # Get data
        yesterday_data = self.get_yesterday_data()
        historical_data = self.get_historical_data()
        
        logger.info(f"Yesterday data: {len(yesterday_data)} records")
        logger.info(f"Historical data: {len(historical_data)} records")
        
        if yesterday_data.empty:
            logger.warning("No data found for yesterday")
            return []
        
        if historical_data.empty:
            logger.warning("No historical data found")
            return []
        
        # Log AM distribution for debugging
        if 'account_manager' in yesterday_data.columns:
            am_distribution = yesterday_data['account_manager'].value_counts()
            logger.info(f"Account Manager distribution: {am_distribution.to_dict()}")
        
        # Keep data at kitchen+brand+source level - DO NOT AGGREGATE
        yesterday_agg = self.calculate_kitchen_aggregates(yesterday_data)
        historical_agg = self.calculate_kitchen_aggregates(historical_data)
        
        logger.info(f"Yesterday aggregated: {len(yesterday_agg)} records")
        logger.info(f"Historical aggregated: {len(historical_agg)} records")
        
        # Calculate baselines
        baseline_metrics = self.calculate_baseline_metrics(historical_agg)
        
        logger.info(f"Baseline metrics: {len(baseline_metrics)} records")
        
        # Detect anomalies
        all_anomalies = []
        all_anomalies.extend(self.detect_revenue_anomalies(yesterday_agg, baseline_metrics))
        all_anomalies.extend(self.detect_volume_anomalies(yesterday_agg, baseline_metrics))
        all_anomalies.extend(self.detect_discount_anomalies(yesterday_agg, baseline_metrics))
        
        logger.info(f"✅ Found {len(all_anomalies)} anomalies")
        return all_anomalies

class MetricsCalculator:
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    def get_dashboard_metrics(self, anomalies: List[AnomalyData]) -> Dict:
        """Calculate key dashboard metrics using order_total as revenue"""
        high_risk = len([a for a in anomalies if a.severity == 'high'])
        medium_risk = len([a for a in anomalies if a.severity == 'medium'])
        
        # Get yesterday's metrics using order_total
        query = """
        SELECT 
            COUNT(DISTINCT CONCAT(kitchen, '-', brand, '-', source)) as total_restaurants,
            AVG(order_total) as avg_revenue,
            SUM(order_total) as total_revenue
        FROM vw_up_only_orders 
        WHERE DATE(date) = :yesterday_date
        AND state = 'Completed'
        AND source IN ('swiggy', 'zomato')
        """
        metrics_data = self.db.execute_query(query, {'yesterday_date': self.yesterday})
        
        total_restaurants = metrics_data.iloc[0]['total_restaurants'] if not metrics_data.empty else 0
        avg_revenue = metrics_data.iloc[0]['avg_revenue'] if not metrics_data.empty else 0
        
        anomaly_rate = (len(anomalies) / total_restaurants * 100) if total_restaurants > 0 else 0
        
        return {
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'avg_gmv': f"₹{avg_revenue:.0f}",
            'anomaly_rate': f"{anomaly_rate:.1f}%",
            'total_anomalies': len(anomalies),
            'total_kitchens': total_restaurants
        }
    
    def get_brand_distribution(self, anomalies: List[AnomalyData]) -> Dict:
        """Get anomaly distribution by brand"""
        brand_stats = {}
        
        for anomaly in anomalies:
            brand = anomaly.brand
            if brand not in brand_stats:
                brand_stats[brand] = {'critical': 0, 'high': 0, 'medium': 0}
            
            if anomaly.severity == 'high':
                brand_stats[brand]['critical'] += 1
            elif anomaly.severity == 'medium':
                brand_stats[brand]['high'] += 1
        
        return brand_stats

    def generate_action_items(self, anomalies: List[AnomalyData]) -> Dict:
        """Generate Account Manager action items based on anomalies"""
        action_items = []
        priorities = {
            'immediate': [],
            'today': [],
            'this_week': [],
            'opportunities': []
        }
        
        # Process each anomaly to generate specific action items
        for anomaly in anomalies:
            action_item = self._create_action_item(anomaly)
            if action_item:
                action_items.append(action_item)
                
                # Categorize actions by priority
                if anomaly.severity == 'high':
                    if anomaly.anomaly_type in ['Revenue Drop', 'High Discounting']:
                        priorities['immediate'].append(self._get_immediate_action(anomaly))
                    else:
                        priorities['today'].append(self._get_daily_action(anomaly))
                else:
                    priorities['this_week'].append(self._get_weekly_action(anomaly))
        
        # Add opportunity items for positive anomalies or improvements
        priorities['opportunities'] = self._get_opportunities(anomalies)
        
        return {
            'action_items': action_items,
            'priorities': priorities,
            'summary': {
                'immediate_count': len(priorities['immediate']),
                'today_count': len(priorities['today']),
                'week_count': len(priorities['this_week']),
                'opportunity_count': len(priorities['opportunities'])
            }
        }
    
    def _create_action_item(self, anomaly: AnomalyData) -> Dict:
        """Create detailed action item for specific anomaly"""
        action_templates = {
            'Revenue Spike': {
                'title': '🚨 URGENT - Revenue Spike Alert',
                'priority': 'urgent' if anomaly.severity == 'high' else 'high',
                'color': '#e74c3c',
                'causes': [
                    'Platform algorithm boost (higher ranking on Zomato/Swiggy)',
                    'Competitor kitchen went offline/out of stock',
                    'Platform running location-specific promotions',
                    'Wrong item pricing updated (too low pricing)',
                    'Other brands from same kitchen also spiking'
                ],
                'actions': [
                    'Check if other brands from same kitchen also spiking',
                    'Verify item pricing on Zomato/Swiggy is correct',
                    'Confirm kitchen capacity across all brands',
                    'Check platform ranking position for this brand',
                    'Monitor competitor availability in delivery radius'
                ]
            },
            'Revenue Drop': {
                'title': '🔻 CRITICAL - Revenue Drop',
                'priority': 'critical',
                'color': '#e74c3c',
                'causes': [
                    'Kitchen operational issues (staff shortage/equipment)',
                    'Competitor kitchen launched in same area',
                    'Platform ranking dropped (lower visibility)',
                    'Stock-outs of popular items',
                    'Platform delivery radius reduced',
                    'Other brands from same kitchen also dropping'
                ],
                'actions': [
                    'Check if all brands from this kitchen are affected',
                    'Verify kitchen operations and staff availability',
                    'Check platform ratings and recent customer reviews',
                    'Analyze new competitor cloud kitchens nearby',
                    'Review item availability and stock status'
                ]
            },
            'High Discounting': {
                'title': '⚠️ HIGH - Excessive Discounting',
                'priority': 'high',
                'color': '#f39c12',
                'causes': [
                    'Wrong discount campaign activated on platform',
                    'Platform auto-discounts triggered incorrectly',
                    'Bulk discount codes being misused',
                    'Account manager error in promotion setup',
                    'Platform promotion conflict (multiple offers stacking)'
                ],
                'actions': [
                    'Login to Zomato/Swiggy partner dashboard immediately',
                    'Check all active promotions and discount campaigns',
                    'Verify if discount is authorized by regional manager',
                    'Calculate margin impact across affected orders',
                    'Disable unauthorized promotions immediately'
                ]
            },
            'Order Volume Spike': {
                'title': '📊 VOLUME - Unusual Order Spike',
                'priority': 'medium',
                'color': '#9b59b6',
                'causes': [
                    'Platform promotional campaigns driving volume',
                    'Competitor kitchen capacity issues',
                    'Weather or local events driving demand',
                    'Platform algorithm favoring this brand',
                    'Viral social media mentions or reviews'
                ],
                'actions': [
                    'Monitor kitchen capacity across all brands',
                    'Verify ingredient stock levels for high-volume items',
                    'Check platform delivery partner availability',
                    'Review order fulfillment times and ratings',
                    'Prepare for sustained higher volumes'
                ]
            }
        }
        
        template = action_templates.get(anomaly.anomaly_type, {
            'title': f'⚠️ {anomaly.anomaly_type}',
            'priority': anomaly.severity,
            'color': '#3498db',
            'causes': ['Unusual pattern detected - requires investigation'],
            'actions': ['Contact operations team for detailed analysis']
        })
        
        return {
            'title': template['title'],
            'kitchen': anomaly.kitchen,
            'brand': anomaly.brand,
            'priority': template['priority'],
            'color': template['color'],
            'details': f"{anomaly.actual} vs expected {anomaly.expected} ({anomaly.deviation})",
            'causes': template['causes'],
            'actions': template['actions'],
            'anomaly_type': anomaly.anomaly_type,
            'severity': anomaly.severity,
            'day_type': anomaly.day_type
        }
    
    def _get_immediate_action(self, anomaly: AnomalyData) -> str:
        """Get immediate action item description"""
        actions = {
            'Revenue Drop': f"Audit {anomaly.kitchen} operations affecting {anomaly.brand}",
            'High Discounting': f"Disable excessive discounts at {anomaly.brand}",
            'Revenue Spike': f"Verify {anomaly.kitchen} revenue spike across all brands"
        }
        return actions.get(anomaly.anomaly_type, f"Investigate {anomaly.brand} at {anomaly.kitchen}")
    
    def _get_daily_action(self, anomaly: AnomalyData) -> str:
        """Get daily action item description"""
        actions = {
            'Revenue Spike': f"Check platform rankings for {anomaly.brand}",
            'Order Volume Spike': f"Monitor capacity at {anomaly.kitchen}",
            'High Discounting': f"Review discount campaigns for {anomaly.brand}"
        }
        return actions.get(anomaly.anomaly_type, f"Monitor {anomaly.brand} performance")
    
    def _get_weekly_action(self, anomaly: AnomalyData) -> str:
        """Get weekly action item description"""
        return f"Optimize {anomaly.brand} strategy at {anomaly.kitchen}"
    
    def _get_opportunities(self, anomalies: List[AnomalyData]) -> List[str]:
        """Generate opportunity items based on anomaly patterns"""
        opportunities = []
        
        # Count different types of anomalies
        revenue_spikes = len([a for a in anomalies if a.anomaly_type == 'Revenue Spike'])
        high_discounts = len([a for a in anomalies if a.anomaly_type == 'High Discounting'])
        volume_spikes = len([a for a in anomalies if a.anomaly_type == 'Order Volume Spike'])
        
        if revenue_spikes > 0:
            opportunities.append(f"{revenue_spikes} brands showing revenue growth (optimize capacity)")
        
        if high_discounts > 0:
            opportunities.append(f"{high_discounts} brands over-discounting (margin optimization)")
        
        if volume_spikes > 0:
            opportunities.append(f"{volume_spikes} brands with volume spikes (scale opportunity)")
        
        # Add general opportunities
        opportunities.extend([
            "Platform ranking improvements for underperforming brands",
            "Cross-brand promotional opportunities",
            "Weekend capacity optimization needed",
            "Standardize discount strategy across all brands"
        ])
        
        return opportunities[:4]  # Limit to top 4 opportunities

# Initialize global objects
db_connection = DatabaseConnection()
anomaly_detector = AnomalyDetector(db_connection)
metrics_calculator = MetricsCalculator(db_connection)

# API Routes
@app.route('/')
def index():
    """Serve the main dashboard"""
    return send_from_directory('static', 'dashboard.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        test_query = "SELECT 1"
        result = db_connection.execute_query(test_query)
        db_status = "connected" if not result.empty else "disconnected"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': db_status,
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/anomalies', methods=['GET'])
def get_anomalies():
    """Get all detected anomalies"""
    try:
        anomalies = anomaly_detector.get_all_anomalies()
        
        # Convert to dict for JSON serialization
        anomalies_dict = []
        for anomaly in anomalies:
            anomalies_dict.append({
                'date': anomaly.date,
                'kitchen': anomaly.kitchen,
                'brand': anomaly.brand,
                'source': anomaly.day_type,  # Source stored in day_type field
                'type': anomaly.anomaly_type,
                'expected': anomaly.expected,
                'actual': anomaly.actual,
                'deviation': anomaly.deviation,
                'severity': anomaly.severity,
                'anomalyType': anomaly.anomaly_type.lower().replace(' ', '_'),
                'impactScore': anomaly.impact_score,
                'accountManager': anomaly.account_manager
            })
        
        return jsonify({
            'success': True,
            'data': anomalies_dict,
            'count': len(anomalies_dict),
            'date': anomaly_detector.yesterday
        })
    
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get dashboard metrics"""
    try:
        anomalies = anomaly_detector.get_all_anomalies()
        metrics = metrics_calculator.get_dashboard_metrics(anomalies)
        
        return jsonify({
            'success': True,
            'data': metrics,
            'date': anomaly_detector.yesterday
        })
    
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/action-items', methods=['GET'])
def get_action_items():
    """Get Account Manager action items"""
    try:
        anomalies = anomaly_detector.get_all_anomalies()
        action_items = metrics_calculator.generate_action_items(anomalies)
        
        return jsonify({
            'success': True,
            'data': action_items,
            'date': anomaly_detector.yesterday
        })
    
    except Exception as e:
        logger.error(f"Error getting action items: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/am-summary', methods=['GET'])
def get_am_summary():
    """Get Account Manager wise anomaly summary"""
    try:
        logger.info("Getting AM summary...")
        anomalies = anomaly_detector.get_all_anomalies()
        logger.info(f"Found {len(anomalies)} anomalies for AM summary")
        
        # Group anomalies by Account Manager
        am_summary = {}
        
        for anomaly in anomalies:
            am = anomaly.account_manager if anomaly.account_manager else 'Unassigned'
            if am not in am_summary:
                am_summary[am] = {
                    'total_anomalies': 0,
                    'critical_count': 0,
                    'medium_count': 0,
                    'revenue_spikes': 0,
                    'revenue_drops': 0,
                    'volume_spikes': 0,
                    'high_discounting': 0,
                    'restaurants_affected': set(),
                    'brands_affected': set()
                }
            
            am_summary[am]['total_anomalies'] += 1
            am_summary[am]['restaurants_affected'].add(anomaly.kitchen)
            am_summary[am]['brands_affected'].add(anomaly.brand)
            
            # Count by severity
            if anomaly.severity == 'high':
                am_summary[am]['critical_count'] += 1
            else:
                am_summary[am]['medium_count'] += 1
            
            # Count by type
            if anomaly.anomaly_type == 'Revenue Spike':
                am_summary[am]['revenue_spikes'] += 1
            elif anomaly.anomaly_type == 'Revenue Drop':
                am_summary[am]['revenue_drops'] += 1
            elif anomaly.anomaly_type == 'Order Volume Spike':
                am_summary[am]['volume_spikes'] += 1
            elif anomaly.anomaly_type == 'High Discounting':
                am_summary[am]['high_discounting'] += 1
        
        # Convert sets to counts for JSON serialization
        for am in am_summary:
            am_summary[am]['restaurants_affected'] = len(am_summary[am]['restaurants_affected'])
            am_summary[am]['brands_affected'] = len(am_summary[am]['brands_affected'])
        
        logger.info(f"AM summary prepared for {len(am_summary)} AMs: {list(am_summary.keys())}")
        
        return jsonify({
            'success': True,
            'data': am_summary,
            'date': anomaly_detector.yesterday
        })
    
    except Exception as e:
        logger.error(f"Error getting AM summary: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug-am', methods=['GET'])
def debug_am():
    """Debug endpoint for Account Manager data"""
    try:
        # Test the up_location_master table
        query = """
        SELECT 
            kitchen,
            brand,
            AM,
            COUNT(*) as record_count
        FROM up_location_master 
        WHERE AM IS NOT NULL AND AM != ''
        GROUP BY kitchen, brand, AM
        ORDER BY kitchen, brand
        LIMIT 10
        """
        am_data = db_connection.execute_query(query)
        
        # Test the join with improved subquery
        join_query = """
        SELECT 
            o.kitchen,
            o.brand,
            o.source,
            COALESCE(am_data.AM, 'Unassigned') as account_manager,
            COUNT(*) as order_count,
            SUM(o.order_total) as total_revenue,
            AVG(o.order_total) as avg_revenue
        FROM vw_up_only_orders o
        LEFT JOIN (
            SELECT DISTINCT kitchen, AM 
            FROM up_location_master 
            WHERE AM IS NOT NULL AND AM != ''
        ) am_data ON TRIM(LOWER(o.kitchen)) = TRIM(LOWER(am_data.kitchen))
        WHERE DATE(o.date) = :yesterday_date
        AND o.state = 'Completed'
        AND o.source IN ('swiggy', 'zomato')
        GROUP BY o.kitchen, o.brand, o.source
        ORDER BY o.kitchen, o.brand, o.source
        LIMIT 10
        """
        join_data = db_connection.execute_query(join_query, {'yesterday_date': anomaly_detector.yesterday})
        
        # Test kitchen+brand aggregation
        agg_query = """
        SELECT 
            o.kitchen,
            o.brand,
            COALESCE(lm.AM, 'Unassigned') as account_manager,
            COUNT(*) as total_orders,
            SUM(o.gmv) as kitchen_brand_gmv,
            DAYNAME(o.date) as day_name
        FROM vw_up_only_orders o
        LEFT JOIN up_location_master lm ON TRIM(LOWER(o.kitchen)) = TRIM(LOWER(lm.kitchen))
        WHERE DATE(o.date) = :yesterday_date
        AND o.state = 'Completed'
        GROUP BY o.kitchen, o.brand, lm.AM, DAYNAME(o.date)
        ORDER BY kitchen_brand_gmv DESC
        LIMIT 5
        """
        agg_data = db_connection.execute_query(agg_query, {'yesterday_date': anomaly_detector.yesterday})
        
        return jsonify({
            'success': True,
            'data': {
                'am_master_sample': am_data.to_dict('records') if not am_data.empty else [],
                'join_test_sample': join_data.to_dict('records') if not join_data.empty else [],
                'aggregation_sample': agg_data.to_dict('records') if not agg_data.empty else [],
                'am_master_count': len(am_data),
                'join_test_count': len(join_data),
                'aggregation_count': len(agg_data),
                'yesterday_date': anomaly_detector.yesterday
            }
        })
    
    except Exception as e:
        logger.error(f"Debug AM endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint for troubleshooting"""
    try:
        # Test database connection
        test_query = "SELECT COUNT(*) as total_records FROM vw_up_only_orders WHERE DATE(date) = :yesterday_date AND state = 'Completed'"
        yesterday_count = db_connection.execute_query(test_query, {'yesterday_date': anomaly_detector.yesterday})
        
        # Get table info
        table_info_query = "DESCRIBE vw_up_only_orders"
        table_info = db_connection.execute_query(table_info_query)
        
        debug_data = {
            'database_connection': 'OK' if db_connection.engine else 'FAILED',
            'yesterday_date': anomaly_detector.yesterday,
            'yesterday_records': int(yesterday_count.iloc[0, 0]) if not yesterday_count.empty else 0,
            'table_columns': len(table_info) if not table_info.empty else 0,
            'config': {
                'host': Config.DB_HOST,
                'database': Config.DB_NAME,
                'user': Config.DB_USER
            }
        }
        
        return jsonify({
            'success': True,
            'data': debug_data,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Connect to database on startup
    logger.info("🚀 Starting Sales Anomaly Detection API...")
    
    if db_connection.connect():
        logger.info("✅ Database connection successful!")
        logger.info(f"📊 Analyzing data for: {anomaly_detector.yesterday}")
        
        # Test anomaly detection on startup
        try:
            anomalies = anomaly_detector.get_all_anomalies()
            logger.info(f"✅ Anomaly detection test: {len(anomalies)} anomalies found")
        except Exception as e:
            logger.warning(f"⚠️ Anomaly detection test failed: {e}")
        
        # Start Flask app
        app.run(
            debug=Config.DEBUG,
            host=Config.HOST,
            port=Config.PORT
        )
    else:
        logger.error("❌ Failed to connect to Azure MySQL database. Exiting...")
        sys.exit(1)
        