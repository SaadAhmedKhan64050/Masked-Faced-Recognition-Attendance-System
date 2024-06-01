import os
import random
import sys
import tkinter as tk
from datetime import datetime
from subprocess import call
from tkinter import *
from tkinter import Entry, Tk, messagebox, ttk


def detect_mask_video_():
    {
        call(["python","detect_mask_video.py"])
    }
    
def ui2_():
    
        pro.withdraw()
        call(["python","ui2.py"])
    
    




if __name__ == '__main__':
    global pro
    pro = Tk()
    pro.title("Face Mask Recognition")
    pro.geometry("480x325")
    pro.resizable(width=False,height=False)
    pro.configure(background='cadet blue')
    Cm = Canvas(width=500, height=320)
    photo_wel = PhotoImage(file='download.png')
    Cm.create_image(0, 0, image=photo_wel, anchor=NW)
    Cm.pack()

    Exe = Button(pro, text="START SYSTEM", width=15,
                     border=1, height=2,
                     font="Arial 0 bold", command=ui2_)
    
    Exe.place(x=0, y=285, height=40, width=500)


    pro.mainloop()