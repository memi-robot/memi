"""Plan test rollouts on RoboMME (blog Section 3.1). At deployment (Section 2.1) the memory policy conditions on its own
memory m_{t-1} from m_0, and the low-level planner provided with RoboMME executes each subgoal the policy chooses.

    __main__.py   the command line, one process per episode
    episode.py    one episode, with a policy call at each decision point
    policies.py   the FullHist policy and the memory policy of Init and MEMI
    capture.py    the object records of every simulator step
    scene.py      a step's text observation
    segments.py   the steps one policy call is shown
    video.py      the episode video
"""
