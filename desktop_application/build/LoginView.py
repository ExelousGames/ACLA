
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path

# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage
import LoginController


OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"assets/frame1")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("508x796")
window.configure(bg = "#FFFFFF")


canvas = Canvas(
    window,
    bg = "#FFFFFF",
    height = 796,
    width = 508,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
canvas.create_text(
    144.0,
    77.0,
    anchor="nw",
    text="Welcome",
    fill="#000000",
    font=("Inter", 16 * -1)
)

canvas.create_text(
    73.0,
    180.0,
    anchor="nw",
    text="Username",
    fill="#000000",
    font=("Inter", 12 * -1)
)

canvas.create_text(
    73.0,
    251.0,
    anchor="nw",
    text="Password",
    fill="#000000",
    font=("Inter", 12 * -1)
)

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: LoginController.get_pokemon_info("dasdsasa"),
    relief="flat"
)


button_1.place(
    x=167.0,
    y=417.0,
    width=174.0,
    height=44.0
)

canvas.create_rectangle(
    73.0,
    272.0,
    435.0,
    314.0,
    fill="#D9D9D9",
    outline="")

entry_image_1 = PhotoImage(
    file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(
    253.0,
    292.5,
    image=entry_image_1
)
entry_1 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_1.place(
    x=88.0,
    y=272.0,
    width=330.0,
    height=39.0
)

canvas.create_rectangle(
    73.0,
    201.0,
    435.0,
    243.0,
    fill="#D9D9D9",
    outline="")

entry_image_2 = PhotoImage(
    file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(
    253.0,
    221.5,
    image=entry_image_2
)
entry_2 = Entry(
    bd=0,
    bg="#FFFFFF",
    fg="#000716",
    highlightthickness=0
)
entry_2.place(
    x=88.0,
    y=201.0,
    width=330.0,
    height=39.0
)
window.resizable(False, False)
window.mainloop()
