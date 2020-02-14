from core.mains.mlconfsetup import mlConfigBasedMain
import sys

# generic ml conf based main, the parts of the program
# that represent a runnable system all implement a main()
# so the same python main with differnet configs 
# can run different programs
if __name__ == "__main__":
    mlConfigBasedMain(sys.argv[1]).main(recursive=True).main()
    
