from core.mains.mlconfsetup import mlConfigBasedMain


# Play games via a SelfPlayWorker implementation
# Report them to somewhere else, while pulling a policy from somewhere else.
# Though the most simple implementation actually trains the policy in here as well
# "Production" level of performance is best reached however by using a distributed setting
# where every server runs as many instances as needed to saturate all resources.
if __name__ == "__main__":
    config = mlConfigBasedMain()
    
    selfplayer = config.selfplayer(recursive=True)
    reporter = config.reporter(recursive=True)
    updater = config.updater(recursive=True)

    selfplayer.selfplay(reporter, updater)

    