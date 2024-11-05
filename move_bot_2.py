#!/usr/bin/env python

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note
from pynput import keyboard
# Navigation works differently in Create 3 and Root robots, so we need to create an instance of the specific robot class here.
robot = Create3(Bluetooth())

keys_pressed = {keyboard.KeyCode.from_char("w"): False, keyboard.KeyCode.from_char("a"): False, keyboard.KeyCode.from_char("s"): False, keyboard.KeyCode.from_char("d"): False}

@event(robot.when_play)
async def play(robot):
  turnSpeed = 8
  moveSpeed = 10
  while(1):
      leftSpeed = 0
      rightSpeed = 0
      if keys_pressed[keyboard.KeyCode.from_char("w")]:
        leftSpeed += moveSpeed
        rightSpeed += moveSpeed
      if keys_pressed[keyboard.KeyCode.from_char("s")]:
        leftSpeed -= moveSpeed
        rightSpeed -= moveSpeed
      if keys_pressed[keyboard.KeyCode.from_char("a")]:
        leftSpeed -= turnSpeed
        rightSpeed += turnSpeed
      if keys_pressed[keyboard.KeyCode.from_char("d")]:
        leftSpeed += turnSpeed
        rightSpeed -= turnSpeed
      await robot.set_wheel_speeds(leftSpeed, rightSpeed)
      print(await robot.get_position())


def on_press(key):
    if (key in keys_pressed.keys()):
      keys_pressed[key] = True
    

def on_release(key):
    if (key in keys_pressed.keys()):
      keys_pressed[key] = False

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

robot.play()