import datetime

enableLogging = False

def logMsg(*txt):
    global enableLogging
    if enableLogging:
        t = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone().isoformat()
        print("["+t+"]", *txt)
    
def setLoggingEnabled(enabled):
    global enableLogging
    enableLogging = enabled
    