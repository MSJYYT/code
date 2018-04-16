import tkinter


class Application(tkinter.Frame):
    count = 0

    def __init__(self, master=None):
        tkinter.Frame.__init__(self, master)

        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.label = tkinter.Label(self, text="hello world")
        self.label.pack(fill="x")

        self.QUIT = tkinter.Button(self)
        self.QUIT["text"] = "退出"
        self.QUIT["foreground"] = "red"
        self.QUIT["command"] = self.quit
        self.QUIT.pack(side=tkinter.RIGHT)

        self.speak = tkinter.Button(self)
        self.speak["text"] = "说话"
        self.speak["foreground"] = "green"
        self.speak["command"] = self.sayHi
        self.speak.pack(side=tkinter.BOTTOM)

        self.scale = tkinter.Scale(self, from_=10, to=40, orient=tkinter.HORIZONTAL, command=self.resize)
        self.scale.pack()

    def sayHi(self):
        Application.count = Application.count + 1
        self.label.config(text="Hello World ! %d" % Application.count)

    def resize(self, ev=None):
        self.label.config(font='Helvetica -%d bold' % self.scale.get())


root = tkinter.Tk()
app = Application(root)
root.geometry('640x360')  # 设置了主窗口的初始大小960x540
root.mainloop()