from tkinter import *
from typing import List, Dict
# from SkyScanner import attributes


def get_multiple_inputs(text: str = "Please enter the following", attributes: List[str] = None) -> Dict[str, str]:
    """
    Create a new window with the text, radio buttons with the options' items and a button to continue.
    When the button is pressed the windows is closed. When the window closes the function returns the index of the the
    item the user chose from the options list.
    :param text: The message to display on top of the inputs.
    :param attributes: The attributes to get the inputs for them.
    :return: A dictionary between the attributes to the given input
    """
    # create the window
    master = Tk()
    # set the title of the window
    master.title("Getting Multiple Inputs")
    # don't allow resizing in the x or y direction
    master.resizable(0, 0)

    # set default options
    if attributes is None:
        attributes = ['att1', 'att2', 'att3']

    inputs = {att: StringVar() for att in attributes}

    # set a label with the msg
    l = Label(master, text=text)
    l.grid(row=0, pady=10, sticky=W + E + S + N)

    # set a radio button for every option
    for i in range(len(attributes)):
        l_att = Label(master, text=attributes[i])
        l_att.grid(row=i+1, column=0,  pady=5, sticky=W)
        input_att = Entry(master, textvariable=inputs[attributes[i]])
        input_att.grid(row=i+1, column=1, pady=5, sticky=W)


    # set a button to continue
    b = Button(master, text="continue", command=master.destroy, width=7, justify=LEFT)
    b.grid(row=len(attributes) + 2, pady=25, sticky=E)

    # run the window
    mainloop()

    # return the chosen value
    return dict((att, inputs[att].get()) for att in attributes)


def get_pref(text: str = "pick your most important option", options=None) -> int:
    """
    Create a new window with the text, radio buttons with the options' items and a button to continue.
    When the button is pressed the windows is closed. When the window closes the function returns the index of the the
    item the user chose from the options list.
    :param text: The message to display on top of the radio-buttons.
    :param options: A list of options for the user to choose his preferences from.
    :return: the index of the option that the user prefer (chose)
    """
    # create the window
    master = Tk()
    # set the title of the window
    master.title("Getting Your Preferences")
    # don't allow resizing in the x or y direction
    master.resizable(0, 0)

    # set default options
    if options is None:
        options = ['option1', 'option2', 'option3']

    # initialize the return value
    v = IntVar()

    # set a label with the msg
    l = Label(master, text=text)
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
    """
    Create {number of options} windows, each time with the text "pick your {i+1}-most important option:",
    and each time gets the user's {i+1} most important option. At the end returns the list of the user's priority.
    :param options: A list of options for the user to choose his priority from.
    :return: A list of the indexes of the options that the user chose in his order of priority, meaning,
             the first int is the index of the option he prioritize the most, and so on.
    """
    # set default options
    if options is None:
        options = ['option1', 'option2', 'option3']
    # return a list of preferences with the number of options
    return [get_pref(text=f"pick your {i+1}-most important option:", options=options) for i in range(len(options))]


def show_text(text: str = "the text to show") -> None:
    """
    Creates a new window with the given text
    :param text: the text to show to the user
    :return: None.
    """
    # create the window
    master = Tk()
    # set the title of the window
    master.title("Showing Text")
    # don't allow resizing in the x or y direction
    master.resizable(0, 0)

    # set a label with the msg
    l = Label(master, text=text)
    l.grid(row=0, pady=10, sticky=W + E + S + N)

    # run the window
    mainloop()


if __name__ == "__main__":
    # results = [attributes[i] for i in get_priority(attributes)]
    # show_text("\n".join(results))
    # from SkyScanner import get_flights, flight_to_string
    # flights = get_flights()[15:20]
    # show_text("The top 5 flights are:\n\n" + "\n\n".join([f"{i+1}. {flight_to_string(flights[i])}" for i in range(len(flights))]))

    get_multiple_inputs(
        attributes=["Country", "Currency", "Locale", "Origin Place", "Destination", "Outbound Date", "Number Of Adults",
                    "Number Of Children", "Cabin Class", "Inbound Date"])

