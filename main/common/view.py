# imports
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

matplotlib.use("TkAgg")

# View is responsible for managing the creation and
# layout of objects on screen
class View(tk.Tk):
    def __init__(self, ctrl, *args, **kwargs):
        self.ctrl = ctrl

        print(ctrl)

        tk.Tk.__init__(self, *args, **kwargs)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (MainPage, HistoryPage, DetailsPage):
            frame = F(container, self, ctrl=self.ctrl)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.tk.call('wm', 'iconphoto', self._w, tk.PhotoImage(file='resources/icon.png'))
        self.wm_title("Ultrasonic spray coating analysis")
        self.show_frame(MainPage)

        self.geometry("1200x500")
        self.resizable(False, False)
        self.mainloop()

    # Switch between frames
    def show_frame(self, controller):
        frame = self.frames[controller]
        frame.tkraise()

# frame
class MainPage(tk.Frame):
    def __init__(self, master, controller, ctrl):

        tk.Frame.__init__(self, master)

        def browse_sample():
            sample_path = tk.filedialog.askdirectory()
            entry.delete(0, 'end')
            entry.insert(tk.END, sample_path)

        def generate_histogram(directory_path):
            diameters = ctrl.process_data(directory_path)
            # mathplotlib
            f = Figure(figsize=(5, 5), dpi=100)
            a = f.add_subplot(111)
            a.hist(diameters, bins=20)
            # toevoegen aan tkinter
            canvas = FigureCanvasTkAgg(f, self)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            # navigation bar toevoegen
            toolbar = NavigationToolbar2Tk(canvas, self)
            toolbar.update()
            canvas._tkcanvas.place(relx=1, rely=0.45, relwidth=0.6, anchor='e')

        browse_button = tk.Button(self, text="Browse...", bg='gray', command=browse_sample)
        browse_button.place(relx=0.125, rely=0.1, relwidth=0.125, anchor='n')

        history_button = tk.Button(self, text="History...", bg='gray',
                                   command=lambda: controller.show_frame(HistoryPage))
        history_button.place(relx=0.275, rely=0.1, relwidth=0.125, anchor='n')

        start_button = tk.Button(self, text="START", bg='gray', command=lambda: generate_histogram(entry.get()))
        start_button.place(relx=0.2, rely=0.4, relwidth=0.15, anchor='n')

        entry = tk.Entry(self, bg='white')
        entry.insert(tk.END, 'C:\\\Path')
        entry.place(relx=0.2, rely=0.2, relwidth=0.3, anchor='n')

        progress_bar = ttk.Progressbar(self, orient='horizontal', mode='determinate')
        progress_bar.place(relx=0.2, rely=0.3, relwidth=0.3, anchor='n')

        hist_label = ttk.Label(self, text="Histogram will appear here.")
        hist_label.place(relx=0.9, rely=0.45, relwidth=0.6, anchor='n')

# frame
class HistoryPage(tk.Frame):
    def __init__(self, master, controller, ctrl):
        tk.Frame.__init__(self, master)

        home_button = tk.Button(self, text="Home...", bg='gray', command=lambda: controller.show_frame(MainPage))
        home_button.place(relx=0.075, rely=0.05, relwidth=0.1, anchor='n')

        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=scrollable_frame, anchor='n')
        canvas.configure(yscrollcommand=scrollbar.set)

        for i in range(50):
            ttk.Label(scrollable_frame, text="Sample scrolling label").pack()

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="right", fill="y", expand=True)

#frame
class DetailsPage(tk.Frame):
    def __init__(self, master, controller, ctrl):
        tk.Frame.__init__(self, master)

        back_button = tk.Button(self, text="Back", bg='gray', command=lambda: controller.show_frame(HistoryPage))
        back_button.pack()
        # back_button.place(relx=0.3, rely=0.1, relwidth=0.3, anchor='n')

        # mathplotlib
        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.hist([1, 1, 1, 1, 1], bins=20)
        # toevoegen aan tkinter
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # navigation bar toevoegen
        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
