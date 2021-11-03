import os

l = []
with open("/homes/cs473/project1/cacm_fullpath.rel", "r") as cacm_file:
    for lines in cacm_file.readlines():
        l.append(lines.rstrip())

final = []
for f in os.listdir("/homes/cs473/project1/cacm100"):
    for i in l:
        if f in i:
            final.append(i)

new_file = open("/homes/dasv/scratch/CS_473/rel_cacm100.rel", "w+")


for line in final:
    new_file.write(f"{line}\n")
