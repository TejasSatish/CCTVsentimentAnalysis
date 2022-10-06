fetch("floorDeets.json",{ mode: 'no-cors'})
    .then(response=>response.json())
    .then(data=>{
        var obj=data;
        console.log(data);
        document.querySelector("#areaName").innerText=obj.floorStatus;
        // document.querySelector("#noOfVehicles").innerText=JSON.stringify(data.noOfVehicles);
        // document.querySelector("#floorStatus").innerText=JSON.stringify(data.floorStatus);
        // document.querySelector("#noOfCounters").innerText=JSON.stringify(data.noOfCounters);
        // document.querySelector("#noOfPeople").innerText=JSON.stringify(data.noOfPeople);
    })
