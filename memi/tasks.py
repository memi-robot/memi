"""The 16 RoboMME tasks."""

#: The tasks by memory suite, in the order of the project page.
SUITES = [("Counting", ["BinFill", "PickXtimes", "SwingXtimes", "StopCube"]),
          ("Permanence", ["VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"]),
          ("Reference", ["PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"]),
          ("Imitation", ["MoveCube", "InsertPeg", "PatternLock", "RouteStick"])]
#: Alphabetical, the order the scripts and the replays run them in.
TASKS = sorted(t for _, ts in SUITES for t in ts)
