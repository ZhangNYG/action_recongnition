import tracker_mosse as TM


with open("39/annotations.txt") as f:

    lines = f.readlines()
    for line in lines :
        param = line.strip().split(" ")
    print(param)

TM.MOSSE()