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


# uint8 command
# geometry_msgs/Twist twist

class KeyCommand():
    def __init__(self):
        self.command = 0
        self.twist = Twist()
        self.HOME = 0
        self.TWIST = 1
        self.PICK = 2
        self.PLACE = 3
        self.FINISH = 4