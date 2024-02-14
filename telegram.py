import requests

def send_telegram_message (message_content):
    TOKEN = "6504288990:AAHfOe84wICn4zbwWcX0iNrRbBK4wSoFHvk"
    chat_id = "6720768632"
    message = message_content
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}%20telah%20hadir"
    print(requests.get(url).json())
