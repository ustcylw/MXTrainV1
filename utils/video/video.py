#! /usr/bin/env python
# coding: utf-8
import os, sys
import cv2
import numpy as np
import imageio


class Video(object):
    
    def __init__(self, mod='r'):
        super().__init__()
        
        self.fps = 0
        self.width:int = 0
        self.height:int = 0
        self.num_frame = 0
        self.fourc = ''
        self.time = 0
        self.time_str = ''
        self.video_file = ''
        # self.reader = None
        # self.writer = None
        self.video = None
        self.win_list = []
        self.mod = mod  # 'r': read, w: write
        self.frame_idx = 0
    
    def init(self, video_file=None, fps=30, width=1080, height=768):
        self.video_file = video_file
        if self.mod == 'r':
            ## read mode
            if video_file is not None:
                if os.path.isdir(video_file):
                    # image folder
                    image_names = os.listdir(video_file)
                    sorted(image_names)
                    self.image_files = [os.path.join(self.video_file, image_name) for image_name in image_names if os.path.splitext(self.video_file)[-1] in ['.jpg', '.png']]
                    
                    self.num_frame = len(self.image_files)
                    image = cv2.imread(self.image_files[0])
                    self.width = image.shape[1]
                    self.height = image.shape[0]
                    self.fps = fps
                elif os.path.isfile(video_file):
                    #  video file
                    self.video = cv2.VideoCapture(self.video_file)
                    self.fps = self.video.get(cv2.CAP_PROP_FPS)                   # 5
                    self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)         # 3
                    self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)       # 4
                    self.num_frame = self.video.get(cv2.CAP_PROP_FRAME_COUNT)     # 7
                    self.fourcc = self.video.get(cv2.CAP_PROP_FOURCC)
                else:
                    # 默认为摄像头
                    self.video = cv2.VideoCapture(0)
                    self.fps = fps
                    ret, frame = self.video.read()
                    assert frame is not None, f'read carama failed!!!'
                    self.width = frame.shape[1]
                    self.height = frame.shape[0]
                    self.num_frame = sys.maxsize
                    self.fourcc = self.video.get(cv2.CAP_PROP_FOURCC)
                self.time = int(self.num_frame / self.fps + 0.5)
                self.time_str = '{}:{}:{}'.format(int(self.time//60//60), int(self.time//60%60), int(self.time%60+0.5))
        elif self.mod == 'w':
            ## write mode
            # fourcc = cv2.VideoWriter_fourcc('I','4','2','0')  # 未压缩的YUV颜色编码，4:2:0色度子采样。兼容性好，但文件非常大。
            # fourcc = cv2.VideoWriter_fourcc('P','I','M','1')  # MPEG-1编码类型。随机访问，灵活的帧率、可变的图像尺寸、定义了I-帧、P-帧和B-帧 、运动补偿可跨越多个帧 、半像素精度的运动向量 、量化矩阵、GOF结构 、slice结构 、技术细节、输入视频格式。
            fourcc = cv2.VideoWriter_fourcc('X','V','I','D')  # MPEG-4编码类型，视频大小为平均值，MPEG4所需要的空间是MPEG1或M-JPEG的1/10，它对运动物体可以保证有良好的清晰度，间/时间/画质具有可调性。
            # # 文件扩展名.ogv:
            # fourcc = cv2.VideoWriter_fourcc('T','H','E','O')  # OGGVorbis，音频压缩格式，有损压缩，类似于MP3等的音乐格式,兼容性差。
            # # 文件扩展名.flv:
            # fourcc = cv2.VideoWriter_fourcc('F','L','V','1')  # FLV是FLASH VIDEO的简称，FLV流媒体格式是一种新的视频格式。由于它形成的文件极小、加载速度极快，使得网络观看视频文件成为可能，它的出现有效地解决了视频文件导入Flash后，使导出的SWF文件体积庞大，不能在网络上很好的使用等缺点。
            self.video = cv2.VideoWriter(self.video_file, fourcc, fps, (width, height))
            self.fourcc = fourcc
            self.fps = fps
            self.width = width
            self.heigth = height
        else:
            raise NotImplementedError(f'Should not reach here!!! mode incorrect!!! mode: {self.mod}')
        
        assert self.video is not None, f'video-{self.mod} is None!!!'
        
        self.release_win_list()
    
    def resize(self, frame):
        return cv2.resize(frame, (int(self.width), int(self.height)))
    
    def read_frame_iter(self):
        assert self.mod == 'r', 'video mod is not read!!!  {self.mod}'
        assert self.video.isOpened(), 'video-reader is not opened!!!'
        
        ret = True
        while ret:
            if os.path.isfile(self.video_file):
                # video file
                ret, frame = self.video.read()
                self.frame_idx = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            elif os.path.isdir(self.video_file):
                # image folder
                if self.frame_idx >= self.num_frame:
                    break
                frame = cv2.imread(self.image_files[self.frame_idx])
                ret = True if frame is not None else False
                self.frame_idx += 1
            else:
                ret, frame = self.video.read()
                self.frame_idx += 1

            if ret and ((self.width != frame.shape[0]) or (self.height != frame.shape[1])):
                frame = self.resize(frame)
            frame = frame[:, :, ::-1]

            yield self.frame_idx, frame
        return None, None

    def __call__(self):
        assert self.mod == 'r', 'video mod is not read!!!  {self.mod}'
        assert self.video.isOpened(), 'video-reader is not opened!!!'
        
        ret = True
        while ret:
            #ret:bool值,是否获取到图片
            ret, frame = self.video.read()
            if ret and ((self.width != frame.shape[0]) or (self.height != frame.shape[1])):
                frame = self.resize(frame)
                frame = frame[:, :, ::-1]
            self.frame_idx = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
            yield self.frame_idx, frame

    def read_frame(self, frame_id=-1):
        assert self.mod == 'r', f'video mod is not read!!!  {self.mod}'
        assert self.video.isOpened(), f'video-{self.mod} is not opened!!!'
        
        if os.path.isfile(self.video_file):
            # video file
            self.frame_idx = self.video.get(cv2.CAP_PROP_POS_FRAMES)  # 1: 获取当前帧位置
            if frame_id >= 0:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                
            ret, frame = self.video.read()
            
            if frame_id >= 0:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_idx)
        elif os.path.isdir(self.video_file):
            # image folder
            assert frame_id < self.num_frame, f'frame id {frame_id} > max num-frame {self.num_frame}!!!'
            frame = cv2.imread(self.image_files[frame_id])
            ret = True if frame is not None else False
        else:
            # carama
            ret, frame = self.video.read()

        if ret and ((self.width != frame.shape[0]) or (self.height != frame.shape[1])):
            frame = self.resize(frame)
        
        # bgr --> rgb
        frame = frame[:, :, ::-1]

        return frame_id, frame

    def write_frame(self, frame):
        assert self.mod == 'w', f'video mod is not read!!!  {self.mod}'
        assert frame is not None, f'frame is None!!!'
        assert self.video.isOpened(), f'video-{self.mod} is not opened!!!'
        
        if (self.width != frame.shape[0]) or (self.height != frame.shape[1]):
            frame = self.resize(frame)
        
        # rgb --> bgr
        frame = frame[:, :, ::-1]
        
        self.video.write(frame)
        
    # def set_WH(self, W=-1, H=-1):
        # if W > 0:
        #     self.video.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        #     self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        #     print(f'w: {self.width}  {W}')
        # if H > 0:
        #     self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        #     self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #     print(f'h: {self.height}  {H}')

    def set_fps(self, fps):
        assert fps > 0, f'{fps} fps is not correct!!!'
        self.video.set(cv2.CAP_PROP_FPS, fps)
        
    def release_win_list(self):
        for win in self.win_list:
            cv2.destroyWindow(win)
        self.win_list = []

    def release(self):
        if self.video is not None:
            self.video.release()
            
        self.release_win_list()


class VideoImageIO(object):
    
    def __init__(self, mod='r'):
        super().__init__()
        
        self.fps = 0
        self.width:int = 0
        self.height:int = 0
        self.num_frame = 0
        self.fourc = ''
        self.time = 0
        self.time_str = ''
        self.video_file = ''
        # self.reader = None
        # self.writer = None
        self.video = None
        self.win_list = []
        self.mod = mod  # 'r': read, w: write
    
    def init(self, video_file=None, fps=30, width=1080, height=768):
        if video_file is not None:
            self.video_file = video_file
            if self.mod == 'r':
                self.video = imageio.get_reader(self.video_file)
            elif self.mod == 'w':
                self.video = imageio.get_writer(self.video_file, fps)
            else:
                raise RuntimeError("Should not be here")

        else:
            # 默认为摄像头
            self.video = cv2.VideoCapture(0)
            self.mod = 'r'
        
        assert self.video is not None, f'video-{self.mod} is None!!!'
        if self.mod == 'r':
            self.fps = self.video.get_meta_data()['fps']                  # 5
            self.width = self.video.get_meta_data()['size'][0]         # 3
            self.height = self.video.get_meta_data()['size'][1]       # 4
            self.num_frame = self.video.count_frames()  #  len(self.video)  ==> inf:float  # get_length()  ==> inf:float  # count_frames()  ==> correct  # get_meta_data()['nframes']     # 7
            self.fourcc = self.video.get_meta_data()['codec']
        
            self.time = int(self.num_frame / self.fps + 0.5) if self.num_frame not in [float("-inf"),float("inf")] else sys.maxsize
            self.time_str = '{}:{}:{}'.format(int(self.time//60//60), int(self.time//60%60), int(self.time%60+0.5))
        elif self.mod == 'w':
            self.fps = fps
            self.width = width
            self.height = height
        else:
            raise RuntimeError("Should not be here")
        
        self.release_win_list()
    
    def resize(self, frame):
        return cv2.resize(frame, (int(self.width), int(self.height)))
    
    def read_frame_iter(self):
        assert self.mod == 'r', 'video mod is not read!!!  {self.mod}'
        assert self.video.isOpened(), 'video-reader is not opened!!!'
        
        while ret:
            #ret:bool值,是否获取到图片
            frame = self.video.get_next_data()
            if (frame is not None) and ((self.width != frame.shape[0]) or (self.height != frame.shape[1])):
                frame = self.resize(frame)
            yield frame, True

    def __call__(self):
        assert self.mod == 'r', 'video mod is not read!!!  {self.mod}'
        assert self.video.isOpened(), 'video-reader is not opened!!!'
        
        while ret:
            #ret:bool值,是否获取到图片
            frame = self.video.get_next_data()
            if (frame is not None) and ((self.width != frame.shape[0]) or (self.height != frame.shape[1])):
                frame = self.resize(frame)
            yield frame, True

    def read_frame(self, frame_id=-1):
        assert self.mod == 'r', f'video mod is not read!!!  {self.mod}'
        assert self.video.isOpened(), f'video-{self.mod} is not opened!!!'

        ret = True
        try:
            frame = self.video.get_data(frame_id)
        except IndexError:
            ret = False
            frame = None
            print(f'IndexError: {frame_id} is out of range!!!')

        if ret and ((self.width != frame.shape[0]) or (self.height != frame.shape[1])):
            frame = self.resize(frame)

        return frame, ret

    def write_frame(self, frame):
        assert self.mod == 'w', f'video mod is not read!!!  {self.mod}'
        assert frame is not None, f'frame is None!!!'
        assert self.video.isOpened(), f'video-{self.mod} is not opened!!!'
        
        if (self.width != frame.shape[0]) or (self.height != frame.shape[1]):
            frame = self.resize(frame)
        self.video.write(frame)
        
    def set_fps(self, fps):
        assert fps > 0, f'{fps} fps is not correct!!!'
        self.video.set(cv2.CAP_PROP_FPS, fps)
        
    def release_win_list(self):
        for win in self.win_list:
            cv2.destroyWindow(win)
        self.win_list = []

    def release(self):
        if self.video is not None:
            self.video.close()
            
        self.release_win_list()



if __name__ == "__main__":
    
    video_file = r'/data2/personal/siammot/siam-mot/videos/001.mp4'
    save_file = r'/data2/personal/siammot/siam-mot/videos/0011.mp4'
    
    reader = Video()
    reader.init(video_file=video_file)
    writer = Video(mod='w')
    writer.init(video_file=save_file, width=1280, height=1280)
        
    iter = reader.read_frame_iter()
    
    for frame_idx, frame in iter:
        if frame is None:
            break
        # print(f'{frame.shape}  {ret}  w: {reader.width}  h: {reader.height}')
        writer.write_frame(frame)
    
    reader.release()
    writer.release()