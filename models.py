import telegram


user_data = {
    "faiz" : False,
    "iza" : False,
    "akmal" : False,
    "gilang" : False,
    "reza" : False,
}

class employee :
    def __init__(self, name) :
        self.name = name

    def update_attendance(self) :
        if (self.name in user_data) : 
            user_data[self.name] = True

    def reset_attendance(self) : 
        if (self.name in user_data) : 
            user_data[self.name] = False

    def send_telegram_msg(self, id) :
        if (user_data[self.name] == False): 
            user_data[self.name] = True
            telegram.send_telegram_message(id)

        