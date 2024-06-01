from tkinter import ttk, Tk
import random
from datetime import datetime
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import Entry
import os
from subprocess import call
import sys

def detect_mask_video_():
    {
        call(["python","detect_mask_video.py"])
        
    }
def att():
    
    pro1 = Tk()
    pro1.title("Attendance")
    pro1.geometry("500x320")
    pro1.resizable(width=False,height=False)
    
    booklist2 = Listbox(pro1, font=('arial', 12), width=150, height=5)
    booklist2.place(x=0, y=0, height=320, width=500)

    booklist2.delete(0, END)  # clear
    file = open("Attendance.csv", 'r')
    lines = file.readlines()
    for line in lines:
        booklist2.insert(END, line)
        file.close()
    pro1.mainloop()
    
    
if __name__ == '__main__':
    pro = Tk()
    pro.title("Attendance")
    pro.geometry("480x280")
    
    pro.resizable(width=False,height=False)
    pro.configure(background='cadet blue')
    Cm = Canvas(width=500, height=320)
    photo_wel = PhotoImage(file='download.png')
    Cm.create_image(0, 0, image=photo_wel, anchor=NW)
    Cm.pack()

    Exe = Button(pro, text="Scan", width=15,
                     border=1, height=2,

                     font="Arial 0 bold", command=detect_mask_video_)
        
    Exe.place(x=10, y=120, height=40, width=160)
    Exe1 = Button(pro, text="Attendance", width=15,
                     border=1, height=2,
                     font="Arial 0 bold", command=att)
   
    Exe1.place(x=317, y=125, height=40, width=160)

    pro.mainloop()