class Pose():
    def __init__(self):
        self.position = type('obj', (object,), {'x':0.0,'y':0.0,'z':0.0})()
        self.orientation = type('obj', (object,), {'x':0.0,'y':0.0,'z':0.0,'w':1.0})()

class PoseArray():
    def __init__(self):
        self.poses = []

class Twist():
    def __init__(self):
        self.linear = type('obj', (object,), {'x':0.0,'y':0.0,'z':0.0})()
        self.angular = type('obj', (object,), {'x':0.0,'y':0.0,'z':0.0})()

class GoalArray():
    def __init__(self):
        self.goals = []

class PoseStamped():
    def __init__(self):
        self.pose = Pose()

class Goal():
    def __init__(self):
        self.id = 0
        self.center = Pose()