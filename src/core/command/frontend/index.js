function formatTimeSince(unixTime, maxMinutes) {
    let ms = Date.now() - unixTime;
    ms /= 1000;
    let minutes = Math.floor(ms / 60);

    if (minutes > maxMinutes) {
        return new Date(unixTime).toISOString();
    }

    let seconds = Math.floor(ms % 60);
    if (minutes > 0) {
        return minutes + "m " + seconds + "s ago";
    } else {
        return seconds + "s ago";
    }
}

function CommandPageModel() {
    let self = this;

    self.password = ko.observable("");

    self.currentHash = ko.observable(location.hash || "#runs/list");
    self.currentHash.subscribe(x => {
        location.hash = x;
    });

    self.showRuns = ko.computed(() => {
        return self.currentHash() === "#runs/list";
    });

    self.selectedRuns = ko.observable("");

    self.showRunsList = ko.computed(() => {
        return self.currentHash().startsWith("#runs/list");
    });

    self.showRunsNew = ko.computed(() => {
        return self.currentHash() === "#runs/new";
    });

    self.showStates = ko.computed(() => {
        return self.currentHash().startsWith("#states");
    });

    self.showNetworks = ko.computed(() => {
        return self.currentHash().startsWith("#networks");
    });

    self.waitingFetches = [];

    self.doPendingFetches = async function() {
        for (let w of self.waitingFetches) {
            await (doAuthFetch(w.url, w.options).then(w.resolve, w.reject));
        }
    };

    function doAuthFetch(url, options) {
        if (!options.headers) {
            options.headers = {};
        }
        options.headers["secret"] = self.password();
        return fetch(url, options);
    }

    self.authFetch = function(url, options) {
        if (self.loggedIn()) {
            return doAuthFetch(url, options);
        } else {
            console.log("Queue up fetch for login");
            let promise = new Promise((resolve, reject) => {
                self.waitingFetches.push({url, options, resolve, reject});
            });
            return promise;
        }
    };

    self.loggedIn = ko.observable(false);

    self.loggedIn.subscribe(x => {
        if (x) {
            self.doPendingFetches();
        }
    });

    self.onLoginPage = ko.computed(() => {
        return !self.loggedIn();
    });

    self.login = async () => {
        let test = await fetch("/password", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8'   
            },
            body: JSON.stringify(self.password())
        });

        if (test.ok) {
            self.loggedIn(true);
        } else {
            alert("Wrong password!");
        }
    };

    self.navRuns = () => {
        self.currentHash("#runs/list");
    };

    self.navStates = () => {
        self.currentHash("#states");
    };

    self.navNetworks = () => {
        self.currentHash("#networks");
    };

    self.navRunsList = () => {
        self.currentHash("#runs/list");
    };

    self.navRunsNew = () => {
        self.currentHash("#runs/new");
    }

    self.selectedRun = ko.observable(null);
    self.runList = ko.observable([]);
    self.networkList = ko.observable([]);
    self.statesList = ko.observable([]);

    self.lastReport = ko.computed(() => {
        let sl = self.statesList();
        if (sl.length > 0) {
            let ts = sl[sl.length - 1].timestamp;
            return formatTimeSince(new Date(ts).getTime(), 60);
        } else {
            return "";
        }
    });

    self.statesCount = ko.computed(() => {
        let sl = self.statesList();
        let cnt = 0;
        for (let s of sl) {
            cnt += s.packageSize;
        }
        return cnt;
    });

    self.statesByWorker = ko.computed(() => {
        let sl = self.statesList();
        let wmap = {};
        for (let s of sl) {
            if (!wmap[s.worker]) {
                wmap[s.worker] = {
                    cnt: 0,
                    lastActive: 0
                };
            }
            wmap[s.worker].cnt += s.packageSize;
            if (wmap[s.worker].lastActive < s.creation) {
                wmap[s.worker].lastActive = s.creation
            }
        }
        let results = [];
        for (let k of Object.keys(wmap)) {
            results.push({
                name: k,
                count: wmap[k].cnt,
                lastActive: new Date(wmap[k].lastActive).toISOString()
            });
        }
        results.sort((a, b) => {
            return new Date(b.lastActive).getTime() - new Date(a.lastActive).getTime();
        });
        results = results.map(x => {
            x.lastActive = formatTimeSince(new Date(x.lastActive).getTime(), 60);
            return x;
        });
        return results;
    });

    self.statesCountByNetwork = ko.computed(() => {
        let sl = self.statesList();
        let nmap = {};
        for (let s of sl) {
            if (!nmap[s.network]) {
                nmap[s.network] = 0;
            }
            nmap[s.network] += s.packageSize;
        }
        return nmap;
    });

    self.selectedRunObject = ko.computed(() => {
        let rlst = self.runList();
        return rlst.find(r => {
            return r.id == self.selectedRun()
        });
    });

    self.newRunYaml = ko.observable("");
    self.newRunName = ko.observable("");

    self.createNewRun = async () => {
        let resp = await self.authFetch("/api/runs/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8'   
            },
            body: JSON.stringify({
                name: self.newRunName(),
                config: self.newRunYaml()
            })
        });

        if (resp.ok) {
            alert("New run created!");
            self.newRunYaml("");
            self.newRunName("");
            self.navRunsList();
        } else {
            alert("Error: " + resp.status);
        }
    };

    self.pullRuns = async () => {
        let runs = await self.authFetch("/api/runs", {
            method: "GET",
        });
        if (runs.ok) {
            let pulled = await runs.json();
            pulled.sort((a, b) => {
                return b.timestamp - a.timestamp
            });
            self.runList(pulled);
        }
    };

    self.pullNetworks = async (forRunId) => {
        let networks = await self.authFetch("/api/networks/list/" + forRunId, {
            method: "GET"
        });
        if (networks.ok) {
            let pulled = await networks.json();
            pulled.sort((a, b) => {
                return a.creation - b.creation;
            });
            for (let i = 0; i < pulled.length; i++) {
                pulled[i].iteration = i + 1;
                pulled[i].download = "/api/networks/download/" + pulled[i].id;
            }
            pulled.sort((a, b) => {
                return b.creation - a.creation;
            });
            self.networkList(pulled);
        }
    };

    self.fancyNetworkList = ko.computed(() => {
        let nets = self.networkList();
        return nets.map(x => {
            x.timestamp = formatTimeSince(x.creation, 60);
            return x;
        });
    });

    self.pullStates = async (forRunId) => {
        let states = await self.authFetch("/api/state/list/" + forRunId, {
            method: "GET"
        });
        if (states.ok) {
            let pulled = await states.json();
            pulled.sort((a, b) => {
                return a.creation - b.creation;
            });
            for (let i = 0; i < pulled.length; i++) {
                pulled[i].timestamp = new Date(pulled[i].creation).toISOString();
                pulled[i].download = "/api/state/download/" + pulled[i].id;
            }
            self.statesList(pulled);
        }
    }

    self.openRun = (run) => {
        self.currentHash("#runs/list/" + run.id);
    }

    self.deleteRun = async () => {
        if (self.selectedRunObject() != null) {
            if (confirm("Really delete the run named " + self.selectedRunObject().name)) {
                let resp = await self.authFetch("/api/runs/" + self.selectedRun(), {
                    method: "DELETE"
                });
                if (resp.ok) {
                    alert("Deleted run");
                } else {
                    alert("Could not delete run. Runs with states cannot be deleted");
                }
                self.navRunsList();
            }
        }
    };

    self.upCounter = 0;

    self.updateForActiveRun = async () => {
        self.upCounter++;
        let runId = self.selectedRun();
        if (runId != null) {
            if (self.upCounter % 20 === 1) {
                console.log("update");
                await self.pullNetworks(runId);
                await self.pullStates(runId);
            } else {
                self.statesList.notifySubscribers();
                let preNets = self.networkList();
                self.networkList([]);
                self.networkList(preNets);
            }
            setTimeout(() => {
                if (self.selectedRun() != null) {
                    self.updateForActiveRun();
                }
            }, 500);
        }
    };

    Sammy(function() {

        this.get("#runs/new", function() {
            self.selectedRun(null);
        });

        this.get("#runs/list", function() {
            console.log("List runs!");
            self.currentHash(location.hash);
            self.pullRuns();
            self.selectedRun(null);
        });

        this.get("#runs/list/:id", function() {
            console.log("Show run", this.params.id);
            self.currentHash(location.hash);
            self.pullRuns().then(async () => {
                self.selectedRun(this.params.id);
                self.updateForActiveRun();
            });
        });

        this.get("#states", function() {
            console.log("Entered states view");
            self.currentHash(location.hash);
            self.selectedRun(null);
        });

        this.get("#networks", function() {
            console.log("Entered networks view");
            self.currentHash(location.hash);
            self.selectedRun(null);
        });

        this.get('', function() { 
            self.pullRuns();
        });
    }).run();
}

let model = new CommandPageModel();

ko.applyBindings(model);