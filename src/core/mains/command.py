from wsgiref.simple_server import make_server
from core.command.app import defineApp

import os
import sys
import json

cfg = {
    "staticPath": os.path.join(os.getcwd(), "core/command/frontend/"),
    "secret": "42"
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

ip = "127.0.0.1"
port = 8042

with make_server(ip, port, defineApp(cfg)) as httpd:
    print("Running command server on port " + str(port))
    httpd.serve_forever()