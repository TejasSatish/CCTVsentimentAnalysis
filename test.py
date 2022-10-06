import json 

noOfpeople=3
camName='lobby'
noOfVehicles=1
floorStatus='busy(minimal)'
noOfCounters="none"
ob={
    "areaName":camName,
    "noOfVehicles":noOfVehicles,
    "floorStatus":floorStatus,
    "noOfCounters":noOfCounters,
    "noOfPeople":noOfpeople
}

json_object=json.dumps(ob,indent=5)
file=open("floorDeets.json","w+")

with file as outfile:
    outfile.write(json_object)