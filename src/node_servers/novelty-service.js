/*

    Has one single API /report/<run> which tracks reported SHAs and then calls command to give the points.

*/

const express = require("express")
const app = express()
const port = 2142
const bodyParser = require('body-parser')
const request = require('request');

// run -> set
const seenPositions = {}

app.use(bodyParser.json());

let pendingPoints = []

function processReport(runId, report) {
    let newPositions = 0

    let positionsForRun = seenPositions[runId] || new Set();

    for (const hash of report.hashes) {
        if (!positionsForRun.has(hash)) {
            newPositions++;
            positionsForRun.add(hash);
        }
    }

    if (report.winner != null) { // null -> draw
        pendingPoints.push({
            player: report.winner,
            points: newPositions,
            run: runId
        });
    }

    seenPositions[runId] = positionsForRun;
}

app.post("/report/:runId", (req, res) => {
    for (const report of req.body) {
        processReport(req.params.runId, report);
    }

    res.sendStatus(200);
});

setInterval(() => {
    const myWork = pendingPoints;
    pendingPoints = []

    if (myWork.length > 0) {
        console.log(`Post ${myWork.length} point assignments!`);

        const byRun = {};

        for (let w of myWork) {
            if (byRun[w.run] == null) {
                byRun[w.run] = []
            }
            byRun[w.run].push(w);
        }

        for (const run of Object.keys(byRun)) {
            const runWork = byRun[run];
            
            request.post("http://127.0.0.1:8042/api/league/reports/" + run, {
                json: runWork.map(x => {
                    return {
                        "p1": x.player,
                        "p2": x.player,
                        "winner": x.points,
                        "run": run,
                        "policy": "FOOBAR"
                    }
                }),
                headers: {
                    "secret": "42"
                }
            });
        }
    }
}, 1000)

app.listen(port, "0.0.0.0", () => console.log(`Novelty service listening on port ${port}`));
