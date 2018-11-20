import random
file = open("test_set.txt","a")

cday=0;
dayslist = ['-','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

file.write("Date,Day,CodedDay,Zone,Weather,Temperature,ActualTime\n")


for date in range(1,31):
    
    cday+=1;
    
    for j in range(1,145):
        temp = random.randint(6,45)
        wea = random.randint(0,47)
        time = random.randint(20,40)

        file.write(str(date)+"-10-2018," + dayslist[cday] + "," + str(cday) + ","+str(j)+","+str(wea)+","+str(temp)+","+str(time)+"\n")
    
    if cday==7 :
        cday=0

file.close()