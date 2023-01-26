from mission import mission
import sys

def main(argv):
    typ = int(argv[0])
    if typ == 1:
       typstr = "nidelva_27_05_21"
    elif typ ==2:
       typstr = "nidelva_08_09_22"
    else: 
       exit()
    mis = mission(typstr)
    mis.emulator()
    mis.mean(version1 = "m", version2 = "tfc")
    mis.fit(end = "test",verbose = True)

if __name__ == "__main__":
   main(sys.argv[1:])