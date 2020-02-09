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

    Sammy(function() {

        this.get("#runs/new", function() {

        });

        this.get("#runs/list", function() {
            console.log("List runs!");
            self.currentHash(location.hash);
            self.pullRuns();
            self.selectedRun(this.params.id);
        });

        this.get("#runs/list/:id", function() {
            console.log("Show run", this.params.id);
            self.currentHash(location.hash);
            self.pullRuns().then(() => {
                self.selectedRun(this.params.id);
            });
        });

        this.get("#states", function() {
            console.log("Entered states view");
            self.currentHash(location.hash);
        });

        this.get("#networks", function() {
            console.log("Entered networks view");
            self.currentHash(location.hash);
        });

        this.get('', function() { 
            self.pullRuns();
        });
    }).run();
}

let model = new CommandPageModel();

ko.applyBindings(model);