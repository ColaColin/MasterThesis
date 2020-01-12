(async function() {

    let addRun = await fetch("/runs/", {
        method: "POST",
        headers: {
            'Content-Type': 'application/json;charset=utf-8'   
        },
        body: JSON.stringify({
            name: "Added run",
            config: "Some config"
        })
    });

    let newRun = "";
    if (addRun.ok) {
        newRun = await addRun.json(); 
        console.log(newRun)
    } else {
        console.log("http error", addRun.status)
    }

    let runs = await fetch("/runs/"+newRun, {
        "method": "GET"
    })
    
    if (runs.ok) {
        console.log(await runs.json());
    } else {
        console.log("Http error: ", runs.status)
    }
    

}())

