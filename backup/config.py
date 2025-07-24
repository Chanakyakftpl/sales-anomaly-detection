# config.py - Configuration Settings
"""
Configuration settings for Sales Anomaly Detection System
"""

class Config:
    # Azure MySQL Database Configuration
    DB_HOST = "wodbinstance1.mysql.database.azure.com"
    DB_NAME = "wodb"
    DB_USER = "wodbuser"
    DB_PASSWORD = "Wopassword$"
    DB_TIMEOUT = 120
    
    # Flask API Configuration
    DEBUG = False
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Anomaly Detection Thresholds
    REVENUE_SPIKE_THRESHOLD = 80.0      # % increase to trigger spike alert
    REVENUE_DROP_THRESHOLD = -40.0      # % decrease to trigger drop alert
    VOLUME_SPIKE_THRESHOLD = 100.0      # % increase in order volume
    HIGH_DISCOUNT_MULTIPLIER = 2.5      # Multiplier for high discount detection
    AOV_SPIKE_THRESHOLD = 60.0          # % increase in average order value
    HIGH_DISCOUNT_THRESHOLD = 0.5       # 50% discount considered high