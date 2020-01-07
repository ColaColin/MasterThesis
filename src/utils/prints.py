enableLogging = False

def logMsg(*txt):
    global enableLogging
    if enableLogging:
        print(*txt)
    
def setLoggingEnabled(enabled):
    global enableLogging
    enableLogging = enabled
    