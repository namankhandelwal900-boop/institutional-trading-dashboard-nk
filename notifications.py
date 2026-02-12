"""
Notification System for Elite Trading App
Supports: Telegram, Email (Placeholder), Logging
"""

import requests
import streamlit as st
from datetime import datetime

class NotificationManager:
    """Manages sending notifications via different channels"""
    
    def __init__(self):
        self.telegram_token = st.session_state.get('telegram_token', '')
        self.telegram_chat_id = st.session_state.get('telegram_chat_id', '')
        
    def send_telegram_message(self, message: str) -> bool:
        """Send message via Telegram Bot"""
        if not self.telegram_token or not self.telegram_chat_id:
            return False
            
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.telegram_chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram Error: {e}")
            return False

    def check_and_send_alerts(self, current_price: float, symbol: str):
        """Check active alerts and trigger notifications"""
        if 'alerts' not in st.session_state:
            return
            
        active_alerts = st.session_state.alerts
        triggered_indexes = []
        
        for i, alert in enumerate(active_alerts):
            if alert['symbol'] == symbol:
                condition_met = False
                
                # Parse condition (e.g., "Price > 100")
                target_price = alert['target_price']
                condition_type = alert['type'] # 'above' or 'below'
                
                if condition_type == 'above' and current_price >= target_price:
                    condition_met = True
                elif condition_type == 'below' and current_price <= target_price:
                    condition_met = True
                    
                if condition_met:
                    msg = f"🚨 *ALERT TRIGGERED*\n\nSymbol: {symbol}\nPrice: {current_price}\nCondition: {condition_type.upper()} {target_price}\nTime: {datetime.now().strftime('%H:%M:%S')}"
                    
                    # Try sending via Telegram
                    sent = self.send_telegram_message(msg)
                    
                    # Show in App
                    st.toast(msg, icon="🚨")
                    if sent:
                        st.toast("Notification sent to Telegram!", icon="📱")
                    
                    triggered_indexes.append(i)
        
        # Remove triggered alerts (reverse order to avoid index shifting)
        for i in sorted(triggered_indexes, reverse=True):
            st.session_state.alerts.pop(i)
