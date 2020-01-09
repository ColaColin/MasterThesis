from core.mains.mlconfsetup import mlConfigBasedMain

# Let a human player play a game vs a learnt instance.
# Configured via a yaml file. See example localplayvs.yaml
if __name__ == "__main__":
    config = mlConfigBasedMain()
    playvs = config.playvs(recursive=True)
    playvs.playVs()
    
