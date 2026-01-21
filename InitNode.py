#!/usr/bin/env python
#coding=utf-8

import time
import numpy as np

import rospy

import zmq
import zmqmsg
import tns

if __name__=="__main__":
    rospy.init_node("initializer")

    timeStep=1./100.
    steps = 5


    # traj_log = Queue()

    rospy.Rate(20)
    
    context = zmq.Context()

    # remoteAddress = "localhost"
    subscriber =context.socket(zmq.REP)
    publisher = context.socket(zmq.REQ)
    subscriber.bind(tns.zmq.Address("*", 33456))
    publisher.bind(tns.zmq.Address("*", 33458))

    ##### initialization #####
    print("prepare to alive")
    identifier, message = zmqmsg.ReceiveMessage(subscriber, timeout=None)

    zmqmsg.SendMessage(publisher, "Experiment", {"task": "init"})
    zmqmsg.SendMessage(publisher, "StartTrial", {"index": -1})
    zmqmsg.SendMessage(publisher, "ShowTarget", {"targetPosition": tuple([3,3,0])})
    sample = np.linspace(0,5,7)
    for i in range(len(sample)-1):
        cur_velo = [0.5,0,0.2]
        zmqmsg.SendMessage(
                publisher,
                    "AvatarInfo",
                    {
                        "avatarPosition": {
                            "z": sample[i],
                            "x": sample[i],
                            "y": 0.0,
                            "avatarRotation": 0.0,
                        },
                        "avatarVelocity": cur_velo,
                    }, timeout=None
                )
        identifer, message = zmqmsg.ReceiveMessage(subscriber, timeout=None)
        time.sleep(0.1)
    zmqmsg.SendMessage(publisher, "end", {})
