
var chart_config = {
    chart: {
        container: "#collapsable-example",

        animateOnInit: true,

        node: {
            collapsable: true
        },
        animation: {
            nodeAnimation: "easeOutBounce",
            nodeSpeed: 700,
            connectorsAnimation: "bounce",
            connectorsSpeed: 700
        }
    },
    nodeStructure: {
        innerHTML: `<textarea style="width: 220px; height: 300px">|Connect4(7,6,4), Turn 1: ░
| 1 | 2 | 3 | 4 | 5 | 6 | 7
|                                
| .   .   .   .   .   .   .  
|                                
| .   .   .   .   .   .   .  
|                                
| .   .   .   .   .   .   .  
|                                
| .   .   .   .   .   .   .  
|                                
| .   .   .   .   .   .   .  
|                                
| .   .   .   .   .   .   .  
|                            </textarea>`,
        children: [
            {
                image: "img/lana.png",
                collapsed: true,
                children: [
                    {
                        image: "img/figgs.png"
                    }
                ]
            },
            {
                image: "img/sterling.png",
                childrenDropLevel: 1,
                children: [
                    {
                        image: "img/woodhouse.png"
                    }
                ]
            },
            {
                pseudo: true,
                children: [
                    {
                        image: "img/cheryl.png"
                    },
                    {
                        image: "img/pam.png"
                    }
                ]
            }
        ]
    }
};

function createNodeStructure(node, depth) {
    if (depth > 5) {
        return null;
    }

    let myContent = "<textarea style='width: 220px; height: 300px'>"+node.state+"</textarea>"

    return {
        innerHTML: myContent,
        collapsed: true,
        children: node.children.map(c => createNodeStructure(c, depth + 1)).filter(x => x != null)
    }
}

// file too big to include in github...
console.log("Fetching...");
fetch("/export_mcts.json").then(x => x.json()).then(x => {
    console.log("Processing tree...");
    var mctsConfig = {
        chart: {
            container: "#collapsable-example",
    
            animateOnInit: true,
    
            node: {
                collapsable: true
            },
            animation: {
                nodeAnimation: "easeOutBounce",
                nodeSpeed: 700,
                connectorsAnimation: "bounce",
                connectorsSpeed: 700
            }
        },
        nodeStructure: createNodeStructure(x, 0)
    };
    console.log("Displaying tree...");
    // this is just way too slow to be of any use whatsoever....
    tree = new Treant( mctsConfig );
});

