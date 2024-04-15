from sshkeyboard import listen_keyboard, listen_keyboard_manual
from time import sleep
from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note
import asyncio
keys_pressed = {'w': False, 'a': False, 's': False, 'd':False}
robot = Create3(Bluetooth())
robotActive = False
@event(robot.when_play)
async def play(robot):
   global robotActive
   print("AE'ERRE")
   robotActive = True
   asyncio.run(listen_keyboard_manual(on_press=on_press, on_release=on_release))

async def on_press(key):
    print(f"{key} pressed")
    if (key in keys_pressed.keys()):
      print("OTHER")
      keys_pressed[key] = True
    await process()
    

async def on_release(key):
    if (key in keys_pressed.keys()):
      keys_pressed[key] = False
    await process()

turnSpeed = 20
moveSpeed = 30

async def process():
    if not robotActive:
       return
    leftSpeed = 0
    rightSpeed = 0
    if keys_pressed['w']:
        leftSpeed += moveSpeed
    rightSpeed += moveSpeed
    if keys_pressed['s']:
        leftSpeed -= moveSpeed
    rightSpeed -= moveSpeed
    if keys_pressed['a']:
        leftSpeed -= turnSpeed
        rightSpeed += turnSpeed
    if keys_pressed['d']:
        leftSpeed += turnSpeed
        rightSpeed -= turnSpeed
    await robot.set_wheel_speeds(leftSpeed, rightSpeed)
    position = await robot.get_position()
    print(position)

robot.play()

print("YOOO")

# listen_keyboard(
#     on_press=on_press,
#     on_release=on_release,
#     )