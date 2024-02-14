from datetime import datetime

def fetch_time_second():
    now = datetime.now()
    return now.second

def fetch_time_minute():
    now = datetime.now()
    return now.minute

def fetch_time_hour():
    now = datetime.now()
    return now.hour

def fetch_date_day():
    now = datetime.now()
    return now.day

def fetch_date_month():
    now = datetime.now()
    return now.month

def fetch_date_year():
    now = datetime.now()
    return now.year

if __name__ == "__main__":
    hour = str(fetch_time_hour())
    minute = str(fetch_time_minute())
    second = str(fetch_time_second())

    day = str(fetch_date_day())
    month = str(fetch_date_month())
    year = fetch_date_year()

    current_time = f"{hour}:{minute}:{second}"
    current_date = f"{year}-{month}-{day}"

    print("Current time:", current_time)
    print("Current date:", current_date)
