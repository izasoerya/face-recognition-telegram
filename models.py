import telegram

user_attendance = {
    "faiz" : False,
    "iza" : False,
    "akmal" : False,
    "gilang" : False,
    "reza" : False,
}
user_attendance_list = list(user_attendance.keys())

class employee :
    def __init__(self, name) :
        self.name = name

    def update_attendance(self) :   ## update attendance when user is detected
        if (self.name in user_attendance) : 
            user_attendance[self.name] = True

    def reset_attendance(self) :    ## reset attendance when day changing
        if (self.name in user_attendance) : 
            user_attendance[self.name] = False

    def send_telegram_msg(self, id) :   ## send telegram message when user is detected
        if (user_attendance[self.name] == False): 
            user_attendance[self.name] = True
            telegram.send_telegram_message(id)
    
    def check_attendance(self) :    ## check attendance status
        return user_attendance[self.name]

        