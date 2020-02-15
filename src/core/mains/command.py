from wsgiref.simple_server import make_server
from core.command.app import defineApp

import os
import sys
import json
import setproctitle

def getCommandConfiguration():
    cfg = {
        "staticPath": os.path.join(os.getcwd(), "core/command/frontend/"),
        "dataPath": "/ImbaKeks/x0/",
        "secret": "42",
        "dbuser": "x0",
        "dbpassword": "x0",
        "dbname": "x0",
        "host": "127.0.0.1",
        "port": 8042
    }

    if "--config" in sys.argv:
        cidx = sys.argv.index("--config") + 1
        cfgPath = sys.argv[cidx];
        print("Opening configuration in file " + str(cfgPath))
        with open(cfgPath, "r") as f:
            cfgTxt = f.read()
            cfg = json.loads(cfgTxt)
            print("Configuration loaded successfully!")
            print(cfgTxt)
            
    return cfg

if __name__ == "__main__":
    cfg = getCommandConfiguration()
    with make_server(cfg["host"], cfg["port"], defineApp(cfg)) as httpd:
        print("Running command server on port " + str(cfg["port"]))
        setproctitle.setproctitle("x0_command")
        httpd.serve_forever()