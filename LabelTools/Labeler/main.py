import sys
import time
import os
import wx
from PIL import Image

import back_end

WIN_WIDTH = 600
WIN_HEIGHT = 850

WIDTH_2_CONVERT = 215
HEIGHT_2_CONVERT = 215

work_dir = "/media/disk1/wsh/label_mission/"
sub_dir_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def WxBitmapToWxImage(myBitmap):
    return wx.ImageFromBitmap(myBitmap)


def WxImageToWxBitmap(myWxImage):
    return myWxImage.ConvertToBitmap()


def PilImageToWxBitmap(myPilImage):
    return WxBitmapToWxImage(PilImageToWxImage(myPilImage))


def PilImageToWxImage(myPilImage):
    x, y = myPilImage.size
    width2c = WIDTH_2_CONVERT
    height2c = HEIGHT_2_CONVERT
    x_s = width2c
    y_s = int((width2c / float(x)) * y)
    if y_s > height2c:
        x_s2 = int((height2c / float(y_s)) * x_s)
        y_s2 = height2c
    else:
        x_s2 = x_s
        y_s2 = y_s
    myPilImage = myPilImage.resize((x_s2, y_s2), Image.ANTIALIAS)
    myWxImage = wx.Image(myPilImage.size[0], myPilImage.size[1])
    myWxImage.SetData(myPilImage.convert('RGB').tobytes())
    return myWxImage


class MyTool(wx.Frame):
    def __init__(self, parent, work_path):
        print('Welcome to Labeling Tool')
        # Window and Panel
        tool_W = 800
        tool_H = 650
        self.img_show_height = 300.0
        self.img_show_width = 450.0
        self.limit_h = 450.0
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title='Labeling Tool', size=(tool_W, tool_H))

        self.bkg = wx.Panel(self, size=wx.Size(tool_W, tool_H))
        self.Center()
        self.SetMaxSize((tool_W, tool_H + 100))
        self.SetMinSize((tool_W, tool_H))

        # Parameters
        self.work_dir = work_path
        print("------->", self.work_dir)
        self.maxTargets = 1000
        self.RecordFile = None
        self.RecordedLines = None
        self.FilesList = None
        self.now_one_idx = 0

        # Create Controls
        self.Button_next = wx.Button(self.bkg, label="Next >")
        self.Button_prev = wx.Button(self.bkg, label="< Prev")

        self.Text_caption = wx.TextCtrl(self.bkg, value='captions', style=wx.TE_MULTILINE | wx.HSCROLL)
        self.Text_label = wx.TextCtrl(self.bkg, value='bathroom', style=wx.TE_PROCESS_ENTER)

        # Bind Events
        self.Button_next.Bind(wx.EVT_BUTTON, self.Next_Image)
        # self.Button_next.Bind(wx.EVT_TEXT_ENTER, self.Next_Image)
        self.Button_prev.Bind(wx.EVT_BUTTON, self.Prev_Image)
        self.Bind(wx.EVT_TEXT_ENTER, self.Next_Image, self.Text_label)

        try:
            pathDir = os.listdir(self.work_dir)
            pathDir.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))
            self.FilesList = pathDir
            self.maxTargets = len(self.FilesList)
            print("maxTargets:", self.maxTargets)
            nowTargetDir = self.work_dir.split('/')[-1]
            nowTargetRecordFile = self.work_dir + '.txt'
            print("nowTargetRecordFile:", nowTargetRecordFile)
            print("++++++++++++++++++++++++++++++++++++")
            print(nowTargetRecordFile)
            if os.path.exists(nowTargetRecordFile):
                print("Restored from early node.")
                RecordFile = open(nowTargetRecordFile, 'r+')
                self.RecordFile = RecordFile
                self.RecordedLines = self.RecordFile.readlines()
                self.RecordedList = [x.split('\n')[0] for x in self.RecordedLines]
                # print(self.RecordedLines)
                # print(self.RecordedList)
                self.now_one_idx = len(self.RecordedList)
                next_one_img_name = self.FilesList[self.now_one_idx]
                print(next_one_img_name)
                next_img_path = self.work_dir + '/' + next_one_img_name
                self.pic_init = wx.Image(next_img_path, wx.BITMAP_TYPE_JPEG)
                bmp_m_w = self.pic_init.GetWidth() * 0.1 * 10
                bmp_m_h = self.pic_init.GetHeight() * 0.1 * 10
                print(bmp_m_w, bmp_m_h)
                rew = self.img_show_width
                reh = self.img_show_width * (bmp_m_h / bmp_m_w)
                if reh > self.limit_h:
                    reh = self.limit_h
                print(rew, reh)
                self.pic_init = self.pic_init.Scale(width=rew, height=reh)
                self.pic_init = self.pic_init.ConvertToBitmap()
                self.image_show = wx.StaticBitmap(self.bkg, bitmap=self.pic_init)
                try:
                    str_cap = back_end.getCaption(next_one_img_name)
                finally:
                    print("Caption get successfully!")
                self.Text_label.Clear()
                self.Text_caption.Clear()
                self.Text_caption.AppendText(str_cap)
            else:
                print("Create New One!")
                RecordFile = open(nowTargetRecordFile, 'w+')
                self.RecordFile = RecordFile
                first_file_name = self.FilesList[0]
                first_file_path = self.work_dir + '/' + first_file_name
                print(first_file_path)
                self.pic_init = wx.Image(first_file_path, wx.BITMAP_TYPE_JPEG)
                bmp_m_w = self.pic_init.GetWidth() * 0.1 * 10
                bmp_m_h = self.pic_init.GetHeight() * 0.1 * 10
                print(bmp_m_w, bmp_m_h)
                rew = self.img_show_width
                reh = self.img_show_width * (bmp_m_h / bmp_m_w)
                if reh >= self.limit_h:
                    reh = self.limit_h
                print(rew, reh)
                self.pic_init = self.pic_init.Scale(width=rew, height=reh)
                self.pic_init = self.pic_init.ConvertToBitmap()
                self.image_show = wx.StaticBitmap(self.bkg, bitmap=self.pic_init)
                try:
                    str_cap = back_end.getCaption(first_file_name)
                finally:
                    print("Caption get successfully!")
                self.Text_label.Clear()
                self.Text_caption.Clear()
                self.Text_caption.AppendText(str_cap)

        finally:
            print("Initiate successfully!")

        # Alignment
        self.vbox_img_cap = wx.BoxSizer(wx.VERTICAL)
        self.vbox_img_cap.Add(self.image_show, proportion=1, flag=wx.EXPAND | wx.CENTER | wx.TOP | wx.LEFT, border=10)
        self.vbox_img_cap.Add(self.Text_caption, proportion=2, flag=wx.EXPAND | wx.ALL, border=10)

        self.hbox_btns = wx.BoxSizer()
        self.hbox_btns.Add(self.Button_prev, proportion=0, flag=wx.TOP | wx.ALIGN_LEFT, border=10)
        self.hbox_btns.Add(self.Button_next, proportion=0, flag=wx.TOP | wx.ALIGN_RIGHT, border=10)

        self.vbox_lab_area = wx.BoxSizer(wx.VERTICAL)
        self.vbox_lab_area.Add(self.Text_label, proportion=1, flag=wx.TOP | wx.ALIGN_CENTER | wx.EXPAND, border=20)
        self.vbox_lab_area.Add(self.hbox_btns, proportion=0, flag=wx.TOP | wx.ALIGN_CENTER, border=10)

        self.hbox_all = wx.BoxSizer()
        self.hbox_all.Add(self.vbox_img_cap, proportion=0, flag=wx.ALL | wx.ALIGN_LEFT, border=10)
        self.hbox_all.Add(self.vbox_lab_area, proportion=1, flag=wx.ALL | wx.ALIGN_CENTER_VERTICAL, border=10)

        self.bkg.SetSizer(self.hbox_all)
        self.Layout()

    def Next_Image(self, event):
        need_rec_img_name = self.FilesList[self.now_one_idx]
        need_rec_label = self.Text_label.GetValue()
        need_rec_one = need_rec_img_name + '/' + need_rec_label + '\n'
        print("RECORD---->", need_rec_one)
        self.RecordFile.write(need_rec_one)
        self.now_one_idx += 1
        next_img_name = self.FilesList[self.now_one_idx]
        print("next img:", next_img_name)
        next_img_path = self.work_dir + '/' + next_img_name
        self.pic_init = wx.Image(next_img_path, wx.BITMAP_TYPE_JPEG)
        bmp_m_w = self.pic_init.GetWidth() * 0.1 * 10
        bmp_m_h = self.pic_init.GetHeight() * 0.1 * 10
        print(bmp_m_w, bmp_m_h)
        rew = self.img_show_width
        reh = self.img_show_width * (bmp_m_h / bmp_m_w)
        if reh >= self.limit_h:
            reh = self.limit_h
        print(rew, reh)
        self.pic_init = self.pic_init.Scale(width=rew, height=reh)
        self.pic_init = self.pic_init.ConvertToBitmap()
        self.image_show.SetBitmap(self.pic_init)

        try:
            str_cap = back_end.getCaption(next_img_name)
        finally:
            print("Caption get successfully!")
        self.Text_label.Clear()
        self.Text_caption.Clear()
        self.Text_caption.AppendText(str_cap)

        # Layouts
        self.image_show.Layout()
        self.Text_caption.Layout()
        self.vbox_img_cap.Layout()
        self.hbox_all.Layout()
        print("Next Image")

    def Prev_Image(self, event):
        print("Prev Image")

    def __del__(self):
        print("------======LOGOUT======------")
        self.RecordFile.close()
        pass


def Main_label_tool_start(event):
    if INPUT_dir.GetValue() not in sub_dir_list:
        print("INVALID INPUT, PLEASE CHOOSE ONE BELOW:\n", sub_dir_list)
    else:
        work_path = work_dir + INPUT_dir.GetValue()
        print(work_path)
        print(work_path)
        win_t = MyTool(bkg_m, work_path)
        win_t.Show()


if __name__ == '__main__':
    MAIN_OF_MAIN_WIDTH = 360
    MAIN_OF_MAIN_HEIGHT = 150
    try:
        app_m = wx.App()
        print()
        win_m = wx.Frame(None, title='Image Label Tool V1.0@ExWang')
        win_m.SetMaxSize((MAIN_OF_MAIN_WIDTH, MAIN_OF_MAIN_HEIGHT))
        win_m.Center()
        bkg_m = wx.Panel(win_m)

        # Create Controls
        INPUT_dir = wx.TextCtrl(bkg_m, value="0", pos=(30, 25), size=(290, 40))
        BT_login = wx.Button(bkg_m, label='Login', pos=(140, 75), size=(70, 30))

        # Bind Events
        BT_login.Bind(wx.EVT_BUTTON, Main_label_tool_start)

        # Frame ops
        win_m.Show()
        app_m.MainLoop()

    finally:

        print('*==== This process is terminated ====*')
