from tkinter import *
from typing import List
from SkyScanner import attributes


def get_pref(msg: str = "pick your most important option", options=None) -> int:
    # create the window
    master = Tk()
    # set the title of the window
    master.title("Getting Your Preferences")
    # change the window size
    max_len = max([len(option) for option in options]+[len(msg)])
    master.geometry(f"{max_len*7}x{(len(options)+3)*35}")
    # don't allow resizing in the x or y direction
    master.resizable(0, 0)

    # set default options
    if options is None:
        options = ['option1', 'option2', 'option3']

    # initialize the return value
    v = IntVar()

    # set a label with the msg
    l = Label(master, text=msg)
    l.grid(row=0, pady=10, sticky=W+E+S+N)

    # set a radio button for every option
    for i in range(len(options)):
        b = Radiobutton(master, text=options[i], variable=v, value=i, width=len(options[i]), justify=LEFT)
        b.grid(row=i+1, pady=5, sticky=W)
        b.config(indicatoron=False)

    # set a button to continue
    b = Button(master, text="continue", command=master.destroy, width=7, justify=LEFT)
    b.grid(row=len(options)+2, pady=25, sticky=E)

    # run the window
    mainloop()

    # return the chosen value
    return v.get()


def get_priority(options=None) -> List[int]:
    # set default options
    if options is None:
        options = ['option1', 'option2', 'option3']
    # return a list of preferences with the number of options
    return [get_pref(msg=f"pick your {i+1}-most important option:", options=options) for i in range(len(options))]


if __name__ == "__main__":
    print(get_priority(attributes))

