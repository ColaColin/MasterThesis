/*
 node.js based server to provide the evaluation services for the MCTS self-play worker.
 As far as this script is concered, it just pushes around binary data, allowing somebody to register work,
  somebody else to checkout work, checkin results and finally letting somebody retrieve work results.
 
 // 1: place work
 POST binary data to /queue/ to get a UUID back, which identifies the task
 
 // 2: process work
 GET /queue/ list pending work IDs. Then use checkout/<ID> to checkout a work item. Pick one at random and allow /checkout to fail with 404 in case somebody else was faster!
 GET /checkout/<ID> checkout work by ID (binary data is returned). If a result is not checked in within 30 seconds (?), allow it to be checked out again and print a warning.
 POST /checkin/<ID> checkin work by ID.

 // 3: retrieve work results
 GET /results/ to list finished work IDs
 GET /results/<ID> to get finished work item binary result data


 This only handles a single run at a time, as the workers need to know by themselves what networks to use, based on the active run!
*/

const express = require("express")
const concat = require('concat-stream');
const uuid = require("uuid")
const app = express()
const port = 4242

// uuid -> binary data that represents the work
const openWork = {}

// uuid -> {work: , timestamp: }
const checkedOutWork = {}

// uuid -> time of creation. Cleared after a few hours.
const workAge = {}

// uuid -> time of completion. Stored for a few hours to prevent duplicate work results.
const completedWork = {}

// uuid -> binary data that represents the finished work
const finishedWork = {}

app.use(function(req, res, next){
    req.pipe(concat(function(data){
      req.body = data;
      next();
    }));
  });

app.post("/queue/:newId", (req, res) => {
    // id of the work package is decided by the requestor. They must use a v4 uuid. That makes the requestor code easier.
    const newId = req.params.newId
    openWork[newId] = req.body
    workAge[newId] = Date.now()

    console.log(new Date(), "Queue work", newId, `Now pending ${Object.keys(openWork).length} work items`);
    res.sendStatus(200)
});

app.get("/queue", (req, res) => {
    const items = Object.keys(openWork);
    items.sort((a, b) => {
        return workAge[a] - workAge[b];
    });
    res.json(items);
});

function sendBuffer(res, buffer) {
    res.writeHead(200, {
        'Content-Type': "application/octet-stream",
        'Content-Length': buffer.length
    });
    res.end(buffer);
}

app.use("/checkout/", (req, res) => {
    const workID = req.url.replace("/", "");
    if (openWork[workID] != null) {
        let oldestWork = Number.MAX_SAFE_INTEGER;
        for (const k of Object.keys(openWork)) {
            if (workAge[k] < oldestWork) {
                oldestWork = workAge[k];
            }
        }
        console.log(new Date(), "Checkout work", workID, `Now pending ${Object.keys(openWork).length} work items. Oldest is ${Math.round((Date.now() - oldestWork) / 1000)}s old!`);
        const workItem = openWork[workID];
        delete openWork[workID];
        checkedOutWork[workID] = {
            work: workItem,
            timestamp: Date.now()
        };

        sendBuffer(res, workItem);
    } else {
        res.sendStatus(404);
    }
});

app.use("/checkin/", (req, res) => {
    const workID = req.url.replace("/", "");
    if (completedWork[workID] == null) {
        let cIn = "Unknown processing time";
        if (checkedOutWork[workID] != null) {
            const procTime = Date.now() - checkedOutWork[workID].timestamp;
            cIn = "Processing time: " + procTime + "ms";
        }
        console.log(new Date(), "Checkin result", workID, cIn);
        finishedWork[workID] = req.body;
    }
    delete openWork[workID];
    delete checkedOutWork[workID];
    completedWork[workID] = Date.now();
    res.sendStatus(200);
});

app.use("/results/", (req, res) => {
    const workID = req.url.replace("/", "");
    if (workID.length > 0 && finishedWork[workID] != null) {
        const workResult = finishedWork[workID];
        delete finishedWork[workID];
        sendBuffer(res, workResult);
    } else {
        res.json(Object.keys(finishedWork));
    }
});

setInterval(() => {

    const FAIL_TIME = 60;

    for (const activeWorkKey of Object.keys(checkedOutWork)) {
        const time = checkedOutWork[activeWorkKey].timestamp;
        if (Date.now() - FAIL_TIME * 1000 > time) {
            const work = checkedOutWork[activeWorkKey].work;
            delete checkedOutWork[activeWorkKey];
            openWork[activeWorkKey] = work;
            console.log(new Date(), "A checkout failed: ", activeWorkKey, `Now pending ${Object.keys(openWork).length} work items`);
        }
    }

    for (const k of Object.keys(completedWork)) {
        if (Date.now() - 7200 * 1000 > completedWork[k]) {
            delete completedWork[k];
        }
    }

    for (const k of Object.keys(workAge)) {
        if (Date.now() - 7200 * 1000 > workAge[k]) {
            delete workAge[k];
        }
    }

}, 1000);

app.listen(port, "0.0.0.0", () => console.log(`Eval manager listening on port ${port}`));