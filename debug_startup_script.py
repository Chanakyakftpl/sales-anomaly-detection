#!/usr/bin/env python3
"""
Minimal test app to check if basic Flask works
"""

print("🚀 Starting minimal test app...")

try:
    from flask import Flask, jsonify
    from config import Config
    
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return jsonify({
            'status': 'success',
            'message': 'Minimal app is working!',
            'config_test': {
                'revenue_critical': Config.REVENUE_IMPACT_CRITICAL,
                'db_host': Config.DB_HOST,
                'port': Config.PORT
            }
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            'status': 'healthy',
            'message': 'Minimal health check OK'
        })
    
    print("✅ Flask app created successfully")
    print(f"🌐 Starting server on http://{Config.HOST}:{Config.PORT}")
    print("📍 Test URLs:")
    print(f"   - http://localhost:{Config.PORT}/")
    print(f"   - http://localhost:{Config.PORT}/health")
    print("🛑 Press Ctrl+C to stop")
    
    # Start the app
    app.run(
        debug=True,
        host=Config.HOST,
        port=Config.PORT,
        use_reloader=False  # Disable reloader to avoid duplicate processes
    )
    
except Exception as e:
    print(f"❌ Error starting minimal app: {e}")
    import traceback
    traceback.print_exc()