import os
from dotenv import load_dotenv

load_dotenv()

def clean_env_value(key, default_value, value_type=str):
    """Clean environment variable by removing inline comments"""
    raw_value = os.getenv(key, default_value)
    clean_value = raw_value.split('#')[0].strip()
    
    if value_type == float:
        return float(clean_value)
    elif value_type == int:
        return int(clean_value)
    elif value_type == bool:
        return clean_value.lower() == 'true'
    else:
        return clean_value

class Config:
    """Smart Business Impact Focused Anomaly Detection"""
    
    # Database Configuration
    DB_HOST = clean_env_value('DB_HOST', 'wodbinstance1.mysql.database.azure.com')
    DB_USER = clean_env_value('DB_USER', 'wodbuser')
    DB_PASSWORD = clean_env_value('DB_PASSWORD', 'Wopassword$')
    DB_NAME = clean_env_value('DB_NAME', 'wodb')
    DB_TIMEOUT = clean_env_value('DB_TIMEOUT', '30', int)
    
    # Flask Configuration
    DEBUG = clean_env_value('DEBUG', 'True', bool)
    HOST = clean_env_value('HOST', '0.0.0.0')
    PORT = clean_env_value('PORT', '5000', int)
    
    # ================================
    # SMART BUSINESS IMPACT THRESHOLDS
    # ================================
    
    # Revenue Impact Tiers - Based on Actual Business Loss
    REVENUE_CRITICAL_LOSS = clean_env_value('REVENUE_CRITICAL_LOSS', '2000', float)  # ₹2,000+ loss = CRITICAL
    REVENUE_HIGH_LOSS = clean_env_value('REVENUE_HIGH_LOSS', '1000', float)          # ₹1,000+ loss = HIGH
    REVENUE_MEDIUM_LOSS = clean_env_value('REVENUE_MEDIUM_LOSS', '500', float)       # ₹500+ loss = MEDIUM
    
    # Volume Impact Tiers
    VOLUME_CRITICAL_LOSS = clean_env_value('VOLUME_CRITICAL_LOSS', '25', int)        # 25+ orders lost = CRITICAL
    VOLUME_HIGH_LOSS = clean_env_value('VOLUME_HIGH_LOSS', '15', int)               # 15+ orders lost = HIGH
    VOLUME_MEDIUM_LOSS = clean_env_value('VOLUME_MEDIUM_LOSS', '8', int)            # 8+ orders lost = MEDIUM
    
    # AOV Impact Tiers
    AOV_CRITICAL_LOSS = clean_env_value('AOV_CRITICAL_LOSS', '100', float)          # ₹100+ per order lost = CRITICAL
    AOV_HIGH_LOSS = clean_env_value('AOV_HIGH_LOSS', '50', float)                   # ₹50+ per order lost = HIGH
    AOV_MEDIUM_LOSS = clean_env_value('AOV_MEDIUM_LOSS', '25', float)               # ₹25+ per order lost = MEDIUM
    
    # ================================
    # PERCENTAGE THRESHOLDS - Secondary Logic
    # ================================
    
    # These are backup thresholds when absolute impact is not significant
    REVENUE_DROP_THRESHOLD = clean_env_value('REVENUE_DROP_THRESHOLD', '-40', float)  # 40%+ drop
    REVENUE_SPIKE_THRESHOLD = clean_env_value('REVENUE_SPIKE_THRESHOLD', '120', float)
    VOLUME_DROP_THRESHOLD = clean_env_value('VOLUME_DROP_THRESHOLD', '-45', float)   # 45%+ drop
    VOLUME_SPIKE_THRESHOLD = clean_env_value('VOLUME_SPIKE_THRESHOLD', '150', float)
    AOV_DROP_THRESHOLD = clean_env_value('AOV_DROP_THRESHOLD', '-35', float)         # 35%+ drop
    AOV_SPIKE_THRESHOLD = clean_env_value('AOV_SPIKE_THRESHOLD', '80', float)
    
    # Discount thresholds - Enhanced with better business logic
    HIGH_DISCOUNT_MULTIPLIER = clean_env_value('HIGH_DISCOUNT_MULTIPLIER', '2.0', float)  # 100%+ increase = alert
    LOW_DISCOUNT_DROP_THRESHOLD = clean_env_value('LOW_DISCOUNT_DROP_THRESHOLD', '-60', float)  # 60%+ drop = alert
    CRITICAL_DISCOUNT_RATE = clean_env_value('CRITICAL_DISCOUNT_RATE', '40', float)      # 40%+ discount = critical
    HIGH_DISCOUNT_RATE = clean_env_value('HIGH_DISCOUNT_RATE', '30', float)            # 30%+ discount = high
    LOW_DISCOUNT_RATE = clean_env_value('LOW_DISCOUNT_RATE', '5', float)               # <5% discount = concerning
    
    # ================================
    # MINIMUM OPERATION SIZE - Only flag meaningful operations
    # ================================
    
    MIN_REVENUE_TO_CONSIDER = clean_env_value('MIN_REVENUE_TO_CONSIDER', '800', float)   # ₹800+ expected
    MIN_ORDERS_TO_CONSIDER = clean_env_value('MIN_ORDERS_TO_CONSIDER', '5', int)        # 5+ expected orders
    MIN_AOV_TO_CONSIDER = clean_env_value('MIN_AOV_TO_CONSIDER', '100', float)          # ₹100+ expected AOV
    
    # Business logic
    HISTORICAL_DAYS = clean_env_value('HISTORICAL_DAYS', '14', int)
    MIN_ORDERS_FOR_DETECTION = clean_env_value('MIN_ORDERS_FOR_DETECTION', '5', int)
    MIN_REVENUE_FOR_DETECTION = clean_env_value('MIN_REVENUE_FOR_DETECTION', '1000', float)
    DEFAULT_AM = clean_env_value('DEFAULT_AM', 'Unassigned')
    
    # ================================
    # SMART SEVERITY ALGORITHM
    # ================================
    
    @classmethod
    def get_smart_revenue_severity(cls, expected_revenue: float, actual_revenue: float) -> tuple:
        """
        Smart algorithm that prioritizes absolute business impact
        Returns: (severity, impact_score, should_flag)
        """
        
        # Skip very small operations
        if expected_revenue < cls.MIN_REVENUE_TO_CONSIDER:
            return 'info', 10, False
        
        # Calculate metrics
        absolute_loss = expected_revenue - actual_revenue
        percentage_drop = ((actual_revenue - expected_revenue) / expected_revenue) * 100
        
        # Skip if it's actually an increase
        if absolute_loss <= 0:
            # Check for significant spike
            if percentage_drop >= cls.REVENUE_SPIKE_THRESHOLD:
                return 'info', 20, True
            return 'info', 10, False
        
        # PRIMARY LOGIC: Absolute Impact Classification
        if absolute_loss >= cls.REVENUE_CRITICAL_LOSS:
            severity = 'critical'
            impact_score = 95 + min(int(absolute_loss / 1000), 20)  # Boost score for larger losses
            should_flag = True
            
        elif absolute_loss >= cls.REVENUE_HIGH_LOSS:
            severity = 'high'
            impact_score = 80 + min(int(absolute_loss / 500), 15)
            should_flag = True
            
        elif absolute_loss >= cls.REVENUE_MEDIUM_LOSS:
            severity = 'medium'
            impact_score = 65 + min(int(absolute_loss / 200), 10)
            should_flag = True
            
        else:
            # SECONDARY LOGIC: Check percentage for smaller absolute losses
            if percentage_drop <= cls.REVENUE_DROP_THRESHOLD:
                # Extreme percentage drop, but small absolute impact
                if percentage_drop <= -60:
                    severity = 'high'
                    impact_score = 75
                elif percentage_drop <= -50:
                    severity = 'medium'
                    impact_score = 60
                else:
                    severity = 'medium'
                    impact_score = 50
                should_flag = True
            else:
                # Neither significant absolute loss nor extreme percentage
                severity = 'info'
                impact_score = 20
                should_flag = False
        
        return severity, impact_score, should_flag
    
    @classmethod
    def get_smart_volume_severity(cls, expected_orders: float, actual_orders: float) -> tuple:
        """Smart volume anomaly detection"""
        
        if expected_orders < cls.MIN_ORDERS_TO_CONSIDER:
            return 'info', 10, False
        
        absolute_loss = expected_orders - actual_orders
        percentage_drop = ((actual_orders - expected_orders) / expected_orders) * 100
        
        if absolute_loss <= 0:
            if percentage_drop >= cls.VOLUME_SPIKE_THRESHOLD:
                return 'info', 20, True
            return 'info', 10, False
        
        # Absolute impact classification
        if absolute_loss >= cls.VOLUME_CRITICAL_LOSS:
            severity = 'critical'
            impact_score = 90 + min(int(absolute_loss / 5), 15)
            should_flag = True
            
        elif absolute_loss >= cls.VOLUME_HIGH_LOSS:
            severity = 'high'
            impact_score = 75 + min(int(absolute_loss / 3), 10)
            should_flag = True
            
        elif absolute_loss >= cls.VOLUME_MEDIUM_LOSS:
            severity = 'medium'
            impact_score = 60 + min(int(absolute_loss / 2), 8)
            should_flag = True
            
        else:
            # Check percentage for smaller losses
            if percentage_drop <= cls.VOLUME_DROP_THRESHOLD:
                if percentage_drop <= -70:
                    severity = 'high'
                    impact_score = 70
                elif percentage_drop <= -60:
                    severity = 'medium'
                    impact_score = 55
                else:
                    severity = 'medium'
                    impact_score = 45
                should_flag = True
            else:
                severity = 'info'
                impact_score = 15
                should_flag = False
        
        return severity, impact_score, should_flag
    
    @classmethod
    def get_smart_aov_severity(cls, expected_aov: float, actual_aov: float, expected_discount_rate: float = 0, actual_discount_rate: float = 0, order_count: int = 0) -> tuple:
        """
        OPTIMIZED AOV anomaly detection that considers discount impact and order patterns
        Returns: (severity, impact_score, should_flag)
        """
        
        if expected_aov < cls.MIN_AOV_TO_CONSIDER:
            return 'info', 10, False
        
        absolute_loss = expected_aov - actual_aov
        percentage_drop = ((actual_aov - expected_aov) / expected_aov) * 100
        
        # Skip positive changes (AOV increases are generally good)
        if absolute_loss <= 0:
            return 'info', 10, False
        
        # SMART FILTER 1: Discount-Adjusted AOV Analysis
        discount_impact_factor = 1.0
        if expected_discount_rate > 0 and actual_discount_rate > 0:
            # Calculate expected AOV drop due to discount increase
            discount_change = actual_discount_rate - expected_discount_rate
            
            # If discount increased significantly, AOV drop is expected
            if discount_change > 5:  # 5+ percentage points increase
                # Reduce the perceived severity since discount explains the AOV drop
                discount_impact_factor = 0.3  # Much lower concern
            elif discount_change > 2:  # 2-5 percentage points increase
                discount_impact_factor = 0.6  # Moderate concern
            elif discount_change < -3:  # Discount decreased but AOV still dropped
                discount_impact_factor = 1.5  # Higher concern - AOV should have increased
        
        # SMART FILTER 2: Order Volume Consideration
        volume_factor = 1.0
        if order_count > 0:
            # If very few orders, AOV can be highly volatile
            if order_count <= 3:
                volume_factor = 0.2  # Very low concern for tiny samples
            elif order_count <= 8:
                volume_factor = 0.5  # Lower concern for small samples
            elif order_count >= 50:
                volume_factor = 1.2  # Higher confidence in large samples
        
        # SMART FILTER 3: Adjusted thresholds for AOV
        # AOV needs much higher absolute losses to be actionable
        adjusted_critical = cls.AOV_CRITICAL_LOSS * 1.5  # ₹150 instead of ₹100
        adjusted_high = cls.AOV_HIGH_LOSS * 1.3  # ₹65 instead of ₹50
        adjusted_medium = cls.AOV_MEDIUM_LOSS * 1.2  # ₹30 instead of ₹25
        
        # Apply discount and volume adjustments
        effective_loss = absolute_loss * discount_impact_factor * volume_factor
        
        # ENHANCED LOGIC: Much stricter AOV flagging
        if effective_loss >= adjusted_critical and percentage_drop <= -25:
            # Only flag as critical if both absolute and percentage are significant
            severity = 'critical'
            impact_score = 75 + min(int(effective_loss / 30), 15)  # Lower base score
            should_flag = True
            
        elif effective_loss >= adjusted_high and percentage_drop <= -20:
            severity = 'high'
            impact_score = 60 + min(int(effective_loss / 20), 10)
            should_flag = True
            
        elif effective_loss >= adjusted_medium and percentage_drop <= -15:
            severity = 'medium'
            impact_score = 45 + min(int(effective_loss / 10), 8)
            should_flag = True
            
        else:
            # Much stricter percentage-only flagging
            if percentage_drop <= -40:  # Only extreme percentage drops
                severity = 'medium'
                impact_score = 40
                should_flag = True
            else:
                # Most AOV fluctuations are not actionable
                severity = 'info'
                impact_score = 15
                should_flag = False
        
        return severity, impact_score, should_flag
    
    @classmethod
    def get_smart_discount_severity(cls, expected_discount_rate: float, actual_discount_rate: float) -> tuple:
        """
        Smart discount anomaly detection
        Returns: (severity, impact_score, should_flag, anomaly_type)
        """
        
        # Skip if no meaningful baseline
        if expected_discount_rate <= 0:
            return 'info', 10, False, 'none'
        
        percentage_change = ((actual_discount_rate - expected_discount_rate) / expected_discount_rate) * 100
        absolute_diff = actual_discount_rate - expected_discount_rate
        
        # HIGH DISCOUNTING - Margin erosion risk
        if actual_discount_rate > expected_discount_rate * cls.HIGH_DISCOUNT_MULTIPLIER:
            if actual_discount_rate >= cls.CRITICAL_DISCOUNT_RATE or percentage_change > 200:
                severity = 'critical'
                impact_score = 95
            elif actual_discount_rate >= cls.HIGH_DISCOUNT_RATE or percentage_change > 150:
                severity = 'high'
                impact_score = 85
            else:
                severity = 'medium'
                impact_score = 70
            
            return severity, impact_score, True, 'high_discounting'
        
        # LOW DISCOUNTING - Competitive risk
        elif actual_discount_rate < expected_discount_rate * (1 + cls.LOW_DISCOUNT_DROP_THRESHOLD/100):
            if actual_discount_rate <= cls.LOW_DISCOUNT_RATE or percentage_change < -70:
                severity = 'high'
                impact_score = 80
            elif actual_discount_rate < 10 or percentage_change < -50:
                severity = 'medium'
                impact_score = 65
            else:
                severity = 'medium'
                impact_score = 55
            
            return severity, impact_score, True, 'low_discounting'
        
        # No significant discount anomaly
        return 'info', 20, False, 'none'
    
    @classmethod
    def format_enhanced_description(cls, expected: float, actual: float, metric_type: str, context: dict = None) -> tuple:
        """Format descriptions showing absolute impact with OPTIMIZED AOV logic"""
        
        absolute_loss = expected - actual
        
        if metric_type == 'revenue':
            if absolute_loss >= cls.REVENUE_MEDIUM_LOSS:
                expected_desc = f"₹{expected:.0f} (expected)"
                actual_desc = f"₹{actual:.0f} (₹{absolute_loss:.0f} loss)"
            else:
                expected_desc = f"₹{expected:.0f} (expected)"
                actual_desc = f"₹{actual:.0f} (actual)"
                
        elif metric_type == 'volume':
            if absolute_loss >= cls.VOLUME_MEDIUM_LOSS:
                expected_desc = f"{expected:.0f} orders (expected)"
                actual_desc = f"{actual:.0f} orders ({absolute_loss:.0f} lost)"
            else:
                expected_desc = f"{expected:.0f} orders (expected)"
                actual_desc = f"{actual:.0f} orders (actual)"
                
        elif metric_type == 'aov':
            # Enhanced AOV description with context
            if context and 'discount_change' in context:
                discount_change = context['discount_change']
                if abs(discount_change) > 2:  # Significant discount change
                    expected_desc = f"₹{expected:.0f} per order (baseline AOV)"
                    if discount_change > 0:
                        actual_desc = f"₹{actual:.0f} per order (discount +{discount_change:.1f}pp explains drop)"
                    else:
                        actual_desc = f"₹{actual:.0f} per order (discount {discount_change:.1f}pp, AOV should be higher)"
                else:
                    # Standard AOV description for significant drops
                    if absolute_loss >= cls.AOV_MEDIUM_LOSS * 1.2:  # Higher threshold
                        expected_desc = f"₹{expected:.0f} per order (expected AOV)"
                        actual_desc = f"₹{actual:.0f} per order (₹{absolute_loss:.0f} less per order)"
                    else:
                        expected_desc = f"₹{expected:.0f} per order (expected AOV)"
                        actual_desc = f"₹{actual:.0f} per order (actual AOV)"
            else:
                # Fallback for no context
                if absolute_loss >= cls.AOV_MEDIUM_LOSS * 1.2:
                    expected_desc = f"₹{expected:.0f} per order (expected AOV)"
                    actual_desc = f"₹{actual:.0f} per order (₹{absolute_loss:.0f} less per order)"
                else:
                    expected_desc = f"₹{expected:.0f} per order (expected AOV)"
                    actual_desc = f"₹{actual:.0f} per order (actual AOV)"
        
        return expected_desc, actual_desc
    
    # ================================
    # LEGACY COMPATIBILITY METHODS
    # ================================
    
    @classmethod
    def get_revenue_severity_by_impact(cls, expected_revenue, actual_revenue, percentage_severity):
        """Legacy compatibility - now uses smart algorithm"""
        severity, _, _ = cls.get_smart_revenue_severity(expected_revenue, actual_revenue)
        return severity
    
    @classmethod
    def get_volume_severity_by_impact(cls, expected_orders, actual_orders, percentage_severity):
        """Legacy compatibility - now uses smart algorithm"""
        severity, _, _ = cls.get_smart_volume_severity(expected_orders, actual_orders)
        return severity
    
    @classmethod
    def get_aov_severity_by_impact(cls, expected_aov, actual_aov, percentage_severity):
        """Legacy compatibility - now uses smart algorithm"""
        severity, _, _ = cls.get_smart_aov_severity(expected_aov, actual_aov)
        return severity
    
    @classmethod
    def should_flag_revenue_drop(cls, expected_revenue, actual_revenue, deviation_pct):
        """Legacy compatibility - now uses smart algorithm"""
        _, _, should_flag = cls.get_smart_revenue_severity(expected_revenue, actual_revenue)
        return should_flag
    
    @classmethod
    def should_flag_volume_drop(cls, expected_orders, actual_orders, deviation_pct):
        """Legacy compatibility - now uses smart algorithm"""
        _, _, should_flag = cls.get_smart_volume_severity(expected_orders, actual_orders)
        return should_flag
    
    @classmethod
    def should_flag_aov_drop(cls, expected_aov, actual_aov, deviation_pct):
        """Legacy compatibility - now uses smart algorithm"""
        _, _, should_flag = cls.get_smart_aov_severity(expected_aov, actual_aov)
        return should_flag
    
    # ================================
    # VALIDATION AND UTILITIES
    # ================================
    
    @classmethod
    def validate_config(cls):
        """Validate configuration"""
        required = ['DB_HOST', 'DB_USER', 'DB_PASSWORD', 'DB_NAME']
        missing = []
        
        for field in required:
            if not getattr(cls, field) or getattr(cls, field).startswith('your-'):
                missing.append(field)
        
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")
        
        return True
    
    @classmethod
    def get_anomaly_config_summary(cls):
        """Get smart algorithm configuration summary with enhanced discount logic"""
        return {
            'smart_absolute_thresholds': {
                'revenue_critical_loss': cls.REVENUE_CRITICAL_LOSS,
                'revenue_high_loss': cls.REVENUE_HIGH_LOSS,
                'volume_critical_loss': cls.VOLUME_CRITICAL_LOSS,
                'volume_high_loss': cls.VOLUME_HIGH_LOSS,
                'aov_critical_loss': cls.AOV_CRITICAL_LOSS,
                'aov_high_loss': cls.AOV_HIGH_LOSS
            },
            'backup_percentage_thresholds': {
                'revenue_drop': cls.REVENUE_DROP_THRESHOLD,
                'volume_drop': cls.VOLUME_DROP_THRESHOLD,
                'aov_drop': cls.AOV_DROP_THRESHOLD
            },
            'discount_thresholds': {
                'high_discount_multiplier': cls.HIGH_DISCOUNT_MULTIPLIER,
                'low_discount_drop_threshold': cls.LOW_DISCOUNT_DROP_THRESHOLD,
                'critical_discount_rate': cls.CRITICAL_DISCOUNT_RATE,
                'high_discount_rate': cls.HIGH_DISCOUNT_RATE,
                'low_discount_rate': cls.LOW_DISCOUNT_RATE
            },
            'minimum_operation_size': {
                'min_revenue': cls.MIN_REVENUE_TO_CONSIDER,
                'min_orders': cls.MIN_ORDERS_TO_CONSIDER,
                'min_aov': cls.MIN_AOV_TO_CONSIDER
            },
            'algorithm_features': [
                'Primary: Absolute business impact classification',
                'Secondary: Extreme percentage drops for small losses',
                'Smart: Skips insignificant operations',
                'Enhanced: Dynamic impact scoring',
                'Business-focused: Revenue loss prioritization',
                'Correct AOV: order_total/order_count calculation',
                'Enhanced Discounts: Both high and low discount detection'
            ]
        }

# Test the algorithm when imported
if __name__ == "__main__":
    print("Smart Algorithm Configuration Loaded Successfully")
    print("Available methods:")
    print("- get_smart_revenue_severity()")
    print("- get_smart_volume_severity()")
    print("- get_smart_aov_severity()")
    print("- get_smart_discount_severity()")
    print("- format_enhanced_description()")