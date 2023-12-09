from re import S
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
import glob
import cv2
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

class StyleFrame:
    TENSORFLOW_HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    def __init__(self,args):
        self.Max_Channel_Intensity = 255.0
        self.ROOT_PATH = args.root_path
        self.FRAME_HEIGHT = 360 #Mmaximum height dimension in pixels. Used for down-sampling the video frames
        self.CLEAR_INPUT_FRAME_CACHE = True
        self.INPUT_FPS = 20 # Rate at which you want to capture frames from the input video
        self.INPUT_VIDEO_NAME = args.input_video
        self.SOURCE_VIDEO_NAME = args.source_video
        self.INPUT_VIDEO_PATH = f'{self.ROOT_PATH}/{self.INPUT_VIDEO_NAME}'
        self.SOURCE_VIDEO_PATH = f'{self.ROOT_PATH}/{self.SOURCE_VIDEO_NAME}'
        self.INPUT_FRAME_DIRECTORY = f'{self.ROOT_PATH}/input_frames'
        self.INPUT_IMAGE_DIRECTORY = f'{self.ROOT_PATH}/input_images'
        if not os.path.exists(self.INPUT_FRAME_DIRECTORY):
            os.makedirs(self.INPUT_FRAME_DIRECTORY)
        
        self.INPUT_FRAME_FILE = '{:0>4d}_frame.png'
        self.INPUT_FRAME_PATH = f'{self.INPUT_FRAME_DIRECTORY}/{self.INPUT_FRAME_FILE}'
        if (args.V2V_flag):
            self.STYLE_REF_DIRECTORY = f'{self.ROOT_PATH}/style_ref'
            self.STYLE_SEQUENCE = []
            self.keyframe_count = args.keyframe_count
        else:
            self.STYLE_REF_DIRECTORY = f'{self.ROOT_PATH}/style_ref_static'
            self.STYLE_SEQUENCE = [0,1,2]
        
        if not os.path.exists(self.STYLE_REF_DIRECTORY):
            os.makedirs(self.STYLE_REF_DIRECTORY)
        
        
        self.STYLE_IMAGE_SEQUENCE = []

        self.OUTPUT_FPS = 20
        self.OUTPUT_VIDEO_NAME = args.output_video
        self.OUTPUT_VIDEO_PATH = f'{self.ROOT_PATH}/{self.OUTPUT_VIDEO_NAME}'
        self.OUTPUT_FRAME_DIRECTORY = f'{self.ROOT_PATH}/output_frames'
        if not os.path.exists(self.OUTPUT_FRAME_DIRECTORY):
            os.makedirs(self.OUTPUT_FRAME_DIRECTORY)
        self.OUTPUT_FRAME_FILE = '{:0>4d}_frame.png'
        self.OUTPUT_IMAGE_FILE = '{}_fast.png'
        self.OUTPUT_FRAME_PATH = f'{self.OUTPUT_FRAME_DIRECTORY}/{self.OUTPUT_FRAME_FILE}'
        self.OUTPUT_IMAGE_PATH = f'{self.OUTPUT_FRAME_DIRECTORY}/{self.OUTPUT_IMAGE_FILE}'

        self.GHOST_FRAME_TRANSPARENCY = 0.1
        self.PRESERVE_COLORS = False

        self.TENSORFLOW_CACHE_DIRECTORY = f'{self.ROOT_PATH}/tensorflow_cache'
        self.hub_module = hub.load(self.TENSORFLOW_HUB_HANDLE)
        self.input_frame_directory = glob.glob(f'{self.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.OUTPUT_FRAME_DIRECTORY}/*')
        self.style_directory = glob.glob(f'{self.STYLE_REF_DIRECTORY}/*')
        self.ref_count = len(self.STYLE_SEQUENCE)
        files_to_be_cleared = self.output_frame_directory
        if self.CLEAR_INPUT_FRAME_CACHE:
            files_to_be_cleared += self.input_frame_directory
        #if (args.V2V_flag):
            #files_to_be_cleared +=self.style_directory
        for file in files_to_be_cleared:
            os.remove(file)
        
        self.input_frame_directory = glob.glob(f'{self.INPUT_FRAME_DIRECTORY}/*')
        self.output_frame_directory = glob.glob(f'{self.OUTPUT_FRAME_DIRECTORY}/*')
        self.style_directory = glob.glob(f'{self.STYLE_REF_DIRECTORY}/*')

        if len(self.input_frame_directory):
            self.frame_width = cv2.imread(self.input_frame_directory[0]).shape[1]


    def source_video_keyframes(self,random=False):
        
        source_video_path=self.SOURCE_VIDEO_PATH
        keyframe_count=self.keyframe_count
        frames = []
        video = cv2.VideoCapture(source_video_path)
        print(source_video_path)
        while True:
            read, frame= video.read()
            if not read:
                break
            frames.append(frame)
        frames = np.array(frames)
        video.release()
        cv2.destroyAllWindows()
        idx = np.round(np.linspace(0, len(frames) - 1, keyframe_count + 1)).astype(int)
        if random:
            random_frames = []
            for i in range(len(idx) - 1):
                start = idx[i]
                end = idx[i + 1]
                random_idx = np.random.randint(start, end - 1)
                # print(random_idx)
                random_frames.append(frames[random_idx])
            final_frames = np.array(random_frames)
        else:
            middle_frames = []
            for i in range(len(idx) - 1):
                start = idx[i]
                end = idx[i + 1]
                middle_idx = (start + end) // 2
                # print(middle_idx)
                middle_frames.append(frames[middle_idx])
            final_frames = np.array(middle_frames)
        style_path= self.STYLE_REF_DIRECTORY+"/{}.jpg"
        for i,img in enumerate(final_frames):
            cv2.imwrite(style_path.format(i), img.astype(np.uint8))
        
        self.STYLE_SEQUENCE=[i for i in range(final_frames.shape[0])]
        self.STYLE_IMAGE_SEQUENCE=[i for i in range(final_frames.shape[0])]
        self.ref_count=len(self.STYLE_SEQUENCE)
        
    def get_input_frames(self):
        if len(self.input_frame_directory):
            print("Using cached input frames")
            return
        vid_obj = cv2.VideoCapture(self.INPUT_VIDEO_PATH)
        frame_interval = np.floor((1.0 / self.INPUT_FPS) * 1000)
        success, image = vid_obj.read()
        if image is None:
            raise ValueError(f"ERROR: Please provide missing video: {self.INPUT_VIDEO_PATH}")
        scale_constant = (self.FRAME_HEIGHT / image.shape[0])
        self.frame_width = int(image.shape[1] * scale_constant)
        image = cv2.resize(image, (self.frame_width, self.FRAME_HEIGHT))
        cv2.imwrite(self.INPUT_FRAME_PATH.format(0), image.astype(np.uint8))

        count = 1
        while success:
            msec_timestamp = count * frame_interval
            vid_obj.set(cv2.CAP_PROP_POS_MSEC, msec_timestamp)
            success, image = vid_obj.read()
            if not success:
                break
            image = cv2.resize(image, (self.frame_width, self.FRAME_HEIGHT))
            cv2.imwrite(self.INPUT_FRAME_PATH.format(count), image.astype(np.uint8))
            count += 1
        self.input_frame_directory = glob.glob(f'{self.INPUT_FRAME_DIRECTORY}/*')
        
        

    def get_style_info(self):
        frame_length = len(self.input_frame_directory)
        style_refs = list()
        resized_ref = False
        style_files = sorted(self.style_directory)
        self.t_const = frame_length if self.ref_count == 1 else np.ceil(frame_length / (self.ref_count - 1))

        # Open first style ref and force all other style refs to match size
        first_style_ref = cv2.imread(style_files.pop(0))
        first_style_ref = cv2.cvtColor(first_style_ref, cv2.COLOR_BGR2RGB)
        first_style_height, first_style_width, _rgb = first_style_ref.shape
        style_refs.append(first_style_ref / self.Max_Channel_Intensity)

        for filename in style_files:
            style_ref = cv2.imread(filename)
            style_ref = cv2.cvtColor(style_ref, cv2.COLOR_BGR2RGB)
            style_ref_height, style_ref_width, _rgb = style_ref.shape
            # Resize all style_ref images to match first style_ref dimensions
            if style_ref_width != first_style_width or style_ref_height != first_style_height:
                resized_ref = True
                style_ref = cv2.resize(style_ref, (first_style_width, first_style_height))
            style_refs.append(style_ref / self.Max_Channel_Intensity)

        if resized_ref:
            print("WARNING: Resizing style images which may cause distortion. To avoid this, please provide style images with the same dimensions")

        self.transition_style_seq = list()
        for i in range(self.ref_count):
            if self.STYLE_SEQUENCE[i] is None:
                self.transition_style_seq.append(None)
            else:
                self.transition_style_seq.append(style_refs[self.STYLE_SEQUENCE[i]])
        print(len(self.transition_style_seq))

    def _trim_img(self, img):
        return img[:self.FRAME_HEIGHT, :self.frame_width]

    def get_output_frames(self):
        self.input_frame_directory = glob.glob(f'{self.INPUT_FRAME_DIRECTORY}/*')
        ghost_frame = None
        for count, filename in enumerate(sorted(self.input_frame_directory)):
            if count % 10 == 0:
                print(f"Output frame: {(count/len(self.input_frame_directory)):.0%}")
            content_img = cv2.imread(filename) 
            content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB) / self.Max_Channel_Intensity
            curr_style_img_index = int(count / self.t_const)
            mix_ratio = 1 - ((count % self.t_const) / self.t_const)
            inv_mix_ratio = 1 - mix_ratio

            prev_image = self.transition_style_seq[curr_style_img_index]
            next_image = self.transition_style_seq[curr_style_img_index + 1]
            
            prev_is_content_img = False
            next_is_content_img = False
            if prev_image is None:
                prev_image = content_img
                prev_is_content_img = True
            if next_image is None:
                next_image = content_img
                next_is_content_img = True
            # If both, don't need to apply style transfer
            if prev_is_content_img and next_is_content_img:
                temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.Max_Channel_Intensity
                cv2.imwrite(self.OUTPUT_FRAME_PATH.format(count), temp_ghost_frame)
                continue
            
            if count > 0:
                content_img = ((1 - self.GHOST_FRAME_TRANSPARENCY) * content_img) + (self.GHOST_FRAME_TRANSPARENCY * ghost_frame)
            content_img = tf.cast(tf.convert_to_tensor(content_img), tf.float32)

            if prev_is_content_img:
                blended_img = next_image
            elif next_is_content_img:
                blended_img = prev_image
            else:
                prev_style = mix_ratio * prev_image
                next_style = inv_mix_ratio * next_image
                blended_img = prev_style + next_style

            blended_img = tf.cast(tf.convert_to_tensor(blended_img), tf.float32)
            expanded_blended_img = tf.constant(tf.expand_dims(blended_img, axis=0))
            expanded_content_img = tf.constant(tf.expand_dims(content_img, axis=0))
            # Apply style transfer
            stylized_img = self.hub_module(expanded_content_img, expanded_blended_img).pop()
            stylized_img = tf.squeeze(stylized_img)

            # Re-blend
            if prev_is_content_img:
                prev_style = mix_ratio * content_img
                next_style = inv_mix_ratio * stylized_img
            if next_is_content_img:
                prev_style = mix_ratio * stylized_img
                next_style = inv_mix_ratio * content_img
            if prev_is_content_img or next_is_content_img:
                stylized_img = self._trim_img(prev_style) + self._trim_img(next_style)

            if self.PRESERVE_COLORS:
                stylized_img = self._color_correct_to_input(content_img, stylized_img)
            
            ghost_frame = np.asarray(self._trim_img(stylized_img))

            temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.Max_Channel_Intensity
            cv2.imwrite(self.OUTPUT_FRAME_PATH.format(count), temp_ghost_frame)
        self.output_frame_directory = glob.glob(f'{self.OUTPUT_FRAME_DIRECTORY}/*')

    def _color_correct_to_input(self, content, generated):
        # image manipulations for compatibility with opencv
        content = np.array((content * self.Max_Channel_Intensity), dtype=np.float32)
        content = cv2.cvtColor(content, cv2.COLOR_BGR2YCR_CB)
        generated = np.array((generated * self.Max_Channel_Intensity), dtype=np.float32)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2YCR_CB)
        generated = self._trim_img(generated)
        # extract channels, merge intensity and color spaces
        color_corrected = np.zeros(generated.shape, dtype=np.float32)
        color_corrected[:, :, 0] = generated[:, :, 0]
        color_corrected[:, :, 1] = content[:, :, 1]
        color_corrected[:, :, 2] = content[:, :, 2]
        return cv2.cvtColor(color_corrected, cv2.COLOR_YCrCb2BGR) / self.Max_Channel_Intensity


    def create_video(self):
        self.output_frame_directory = glob.glob(f'{self.OUTPUT_FRAME_DIRECTORY}/*')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(self.OUTPUT_VIDEO_PATH, fourcc, self.OUTPUT_FPS, (self.frame_width, self.FRAME_HEIGHT))

        for count, filename in enumerate(sorted(self.output_frame_directory)):
            if count % 10 == 0:
                print(f"Saving frame: {(count/len(self.output_frame_directory)):.0%}")
            image = cv2.imread(filename)
            video_writer.write(image)

        video_writer.release()
        print(f"Style transfer complete! Output at {self.OUTPUT_VIDEO_PATH}")

    def run(self):
        keyframe_count=5
        print("Getting Source Video Key frames")
        self.source_video_keyframes(random=False)
        print("Getting input frames")
        self.get_input_frames()
        print("Getting style info")
        self.get_style_info()
        print("Getting output frames")
        self.get_output_frames()
        print("Saving video")
        self.create_video()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for Video to Video Style Transfer using Arbitrary Image Stylization")
    parser.add_argument("--root_path", dest='root_path', default=".",
                        help="The directory of your data")
    parser.add_argument("--input_video", dest='input_video', default="short.mov",
                        help="Input Video Name to which you want to apply style")
    parser.add_argument("--style_source_video", dest="source_video", default="source_vid.mov",
                        help="Source video to extract style from")
    parser.add_argument("--V2V_flag", default=True, dest='V2V_flag', type=bool,
                        choices=(True, False),
                        help="Do you want to extract styles from a source video")
    parser.add_argument('--output_video', dest='output_video', default="output_video.mp4",
                        help='Output Video Name')
    parser.add_argument("--keyframe_count", dest='keyframe_count', default=5, type=int,
                        help="Number of style images to extyract from source video")
    
    # parser.add_argument('--decoder_type', dest='decoder_type', default='bernoulli', type=str,
    #                     help='Type of your decoder', choices=('bernoulli', 'gaussian'))
    # parser.add_argument("--Nz", default=20, type=int,
    #                     help="Nz (dimension of the latent code)")

    args = parser.parse_args()
    StyleFrame(args).run()