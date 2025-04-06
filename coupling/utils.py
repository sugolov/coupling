from datetime import datetime

def timestamp(*args):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    label = " ".join(map(str, args))
    print(str(current_time) + ": " + label, flush=True)