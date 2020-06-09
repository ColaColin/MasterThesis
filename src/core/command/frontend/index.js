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

function formatTimeCost(cost) {
    if (typeof cost === "number") {
        return Math.floor(cost / 1000) + "s";
    } else {
        return cost;
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
    self.statsList = ko.observable([]);

    self.displayAll = () => {
        self.showAllPlayers(true)
    };

    self.displayLimited = () => {
        self.showAllPlayers(false);
    };

    self.showAllPlayers = ko.observable(false);
    self.playersList = ko.observable([]);
    self.players = ko.computed(() => {
        let plst = self.playersList();
        return plst.slice(0, self.showAllPlayers() ? 999999 : 3).map(x => {
            return {
                id: x[0].split("-")[0],
                wins: x[3][0],
                losses: x[3][1],
                draws: x[3][2],
                rating: Math.round(x[1]),
                parameters: x[2],
                generation: "G"+x[4]
            }
        });
    });

    self.matches = ko.observable([]);
    self.matchesDisplay = ko.computed(() => {
        let lst = self.matches();
        return lst.map(x => {
            return {
                p1: x[0].split("-")[0],
                p2: x[1].split("-")[0],
                p1Win: x[2] == 1,
                p2Win: x[2] == 0,
                rating: x[3],
                time: formatTimeSince(x[4], 60)
            }
        });
    });

    self.loadReportsAscii = async function() {
        let runs = await self.authFetch("/api/insight/" + self.shownReportId(), {
            method: "GET",
        });
        if (runs.ok) {
            let insightList = await runs.json();
            self.reportsAscii(insightList);
        } else {
            console.log(runs);
            self.reportsAscii(["Error loading!"]);
        }
    };

    self.reportsAscii = ko.observable(["A", "B", "CC"]);

    self.shownReportFrame = ko.observable(0);

    self.shownReportFrame.subscribe(() => {
        setTimeout(() => {
            self.loadReportsAscii()
        }, 1);
    });

    self.shownReportId = ko.computed(() => {
        let state = self.statesList()[self.shownReportFrame()];
        if (state) {
            return state.id;
        } else {
            return "unknown";
        }
    });

    self.nextReportFrame = function() {
        let n = (self.shownReportFrame() + 1) % self.statesList().length;
        self.shownReportFrame(n);
    };

    self.prevReportFrame = function() {
        let n = self.shownReportFrame() - 1;
        if (n < 0) {
            n = self.statesList().length - 1;
        }
        self.shownReportFrame(n);
    };

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
            const networkKey = s.network || "initial";
            if (!nmap[networkKey]) {
                nmap[networkKey] = 0;
            }
            nmap[networkKey] += s.packageSize;
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
    self.newRunSha = ko.observable("");
    self.newRunName = ko.observable("");

    self.createNewRun = async () => {
        let resp = await self.authFetch("/api/runs/", {
            method: "POST",
            headers: {
                'Content-Type': 'application/json;charset=utf-8'   
            },
            body: JSON.stringify({
                name: self.newRunName(),
                config: self.newRunYaml(),
                sha: self.newRunSha()
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

    self.pullLeague = async (forRunId) => {
        let players = await self.authFetch("/api/league/players/" + forRunId, {
            method: "GET"
        });
        if (players.ok) {
            let pulled = await players.json();
            self.playersList([]);
            self.playersList(pulled);
        }
        let m = await self.authFetch("/api/league/matches/"+forRunId, {
            method: "GET"
        });
        if (m.ok) {
            self.matches([]);
            self.matches(await m.json());
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

    self.runCost = ko.computed(() => {
        const nets = self.networkList().slice();
        const states = self.statesCountByNetwork();

        // iteration -> cost
        const result = {};

        if (nets.length === 0 || nets[0].frametime == null) {
            return result;
        }

        nets.sort((a, b) => {
            return a.iteration - b.iteration;
        });

        let cost = nets[0].frametime * states["initial"];

        result[0] = formatTimeCost(cost);

        for (let i = 0; i < nets.length; i++) {
            const net = nets[i];
            if (net.frametime == null) {
                break;
            }

            cost += net.frametime * states[net.id];
            result[i + 1] = formatTimeCost(cost);
        }

        return result
    });

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

    self.pullStats = async (forRunId) => {
        let stats = await self.authFetch("/api/stats/" + forRunId, {
            method: "GET"
        });

        if (stats.ok) {
            let pulled = await stats.json();
            pulled.sort((a, b) => {
                return b.iteration - a.iteration;
            });
            self.statsList(pulled);
        }
    };

    self.numInStats = ko.computed(() => {
        let cnt = 0;
        let lst = self.statsList();
        for (let i = 0; i < lst.length; i++) {
            cnt += lst[i].played_states;
        }
        return cnt;
    });

    self.numDistinct = ko.computed(() => {
        let cnt = 0;
        let lst = self.statsList();
        for (let i = 0; i < lst.length; i++) {
            cnt += lst[i].new_states;
        }
        return cnt;
    });

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
            if (self.upCounter % 10 === 1) {
                console.log("update");
                await self.pullLeague(runId);
                await self.pullNetworks(runId);
                await self.pullStates(runId);
                await self.pullStats(runId);
            } else {
                self.statesList.notifySubscribers();
                
                let preNets = self.networkList();
                self.networkList([]);
                self.networkList(preNets);

                let preStats = self.statsList();
                self.statsList([])
                self.statsList(preStats);
            }
            setTimeout(() => {
                if (self.selectedRun() != null) {
                    self.updateForActiveRun();
                }
            }, 1000);
        }
    };

    Sammy(function() {

        this.get("#runs/new", function() {
            self.selectedRun(null);
            self.shownReportFrame(0);
            self.playersList([]);
            self.matches([]);
        });

        this.get("#runs/list", function() {
            console.log("List runs!");
            self.playersList([]);
            self.matches([]);
            self.shownReportFrame(0);
            self.currentHash(location.hash);
            self.pullRuns();
            self.selectedRun(null);
        });

        this.get("#runs/list/:id", function() {
            console.log("Show run", this.params.id);
            self.shownReportFrame(0);
            self.currentHash(location.hash);
            self.upCounter = 0;
            self.pullRuns().then(async () => {
                self.selectedRun(this.params.id);
                await self.updateForActiveRun();
                self.loadReportsAscii();
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