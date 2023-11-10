# Image processing for C3VD dataset 
#
# Author: Sierra Bonilla
# Date: 27-10-23

import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import os
import glob
import shutil
from moviepy.editor import VideoFileClip, concatenate, vfx

def read_pose(line):
  """
  Reshape line of poses text file into numpy matrix of 4x4.

  :param
      line (): A line in the text file.

  :return
      matrix (np.array): A 4x4 numpy matrix.
  """
  # Convert the line into a list of floats
  flattened_matrix = list(map(float, line.split(',')))

  # Reshape the flattened_matrix to a 4x4 matrix
  matrix = np.array(flattened_matrix).reshape(4,4)

  return matrix

def extract_camera_params(matrix):
  """
  Extract camera position and direction from transformation matrix.

  :param
      matrix (np.array): A 4x4 numpy array.

  :return
      camera_origin (np.array): A 1x3 numpy array with the x,y,z coordinate of camera position.
      camera_direction (np.array): A 1x3 numpy array with the rotation direction.
  """
  # Camera origin (position) given by translation component
  camera_origin = matrix[:3, 3]

  # Camera direction given by opposite of third column of the matrix
  camera_direction = -matrix[:3, 2]

  return camera_origin, camera_direction

def create_mask_from_color_image(image, plotting=False):
  """
  Extract binary mask from one of the images.

  :param
      image (cv2.image): An image from the C3VD dataset.

  :return
      mask (np.array): A numpy array dimension same as image where nonzero pixels are set to 1.
  """
  # Convert image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Create a binary mask where nonzero pixels are set to 1
  mask = np.where(gray_image > 0, 1, 0)

  if plotting:
    plt.imshow(mask,cmap='gray')
    plt.title('Mask')
    plt.show()

  return mask

def apply_mask_to_image(image, mask, plotting=False):
  """
  Apply the binary mask to the color image.

  :param
      image (cv2.image): An image from the C3VD dataset.
      mask (np.array): A numpy array dimension same as image where nonzero pixels are set to 1.

  :return
      mask_image (np.array): Color image without the black edges.
  """
  # Duplicate the mask to have the same shape as the color image
  mask_rgb = np.stack([mask]*3, axis=-1)

  # Apply the mask to the color image
  masked_image = image*mask_rgb

  if plotting:
    plt.imshow(cv2.cvtColor(masked_image.astype('uint8'), cv2.COLOR_BGR2RGB))
    plt.title('Masked Image')
    plt.show()

  return masked_image.astype('uint8')

def get_dataset(directory, resize):
    """
    Gets processed images into numpy array of dimensions [num_frames, num_pixels_in_frames]

    :param
        directory (string): string to directory
        mask (): Mask where all pixels are nonzero (use the same one for each image)
        resize (list)
    """

    # Initialize image names
    png_file_paths = []

    # For every image, append file path name to list
    for file_path in glob.glob(os.path.join(directory, '*.png')):
        png_file_paths.append(file_path)

    # Sort frames on file name by number 
    frames_sorted = sorted(png_file_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Make a mask out of first image
    c_image = cv2.imread(frames_sorted[0])
    mask = create_mask_from_color_image(c_image, plotting=False)
    mask = np.uint8(mask)

    # Define resize 
    mask = cv2.resize(mask, (resize))
  
    # Create boolean mask once
    mask2 = mask != 0

    # Using list comprehension to process all frames and get the dataset
    processed_dataset = [
            apply_mask_to_image(
                cv2.resize(cv2.imread(frame, cv2.IMREAD_COLOR), (resize)),  # Resizing the image first
                mask,
                plotting=False
            )[mask2].reshape(-1)
            for frame in frames_sorted
        ]

    return np.stack(processed_dataset, axis=0)

def process_all_images(input_path, output_path, max_num=1000, start_idx=0):
    """
    only works on jpg or png atm 
    usage: process_all_images(input_path='data/C3VD/seq3', output_path='data/C3VD/seq3_processed')

    :param
        input_path (string): path to input images
        output_path (string): path to output images
        max_num (int): maximum number of images to process
        start_idx (int): index to start at

    :return 
        None
    """
    images = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and (f.endswith('.png') or f.endswith('.jpg'))]
    images = sorted(images, key=lambda x: int(os.path.basename(x).split('.')[0]))

    # if there are no images, flag error!
    if not images:
        print('Unable to find any png or jpg images in the path provided!')
        return

    # If output path doesn't exist - make one
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
       shutil.rmtree(output_path)
       os.makedirs(output_path)
    
    try:

        # Loop through and process images
        for index in range(len(images)):

            if index == max_num + start_idx:
                break

            # check if file exists: 
            if os.path.exists(images[index]):

                if index >= start_idx:
                    image = cv2.imread(images[index])

                    # check where the first column non zero is 
                    first_layer = image[:, 0, :]
                    for idx, pixel in enumerate(first_layer):
                        max_val = max(pixel)
                        if max_val != 0:
                            break
                        
                    processed_image = image[:, idx:(image.shape[1]-idx)]
                    file_path = output_path + '/' + os.path.basename(images[index])
                    cv2.imwrite(file_path, processed_image)
            else:
               print(f"File {images[index]} does not exist.")
               return

        print(f'All images saved to {output_path}')

    except Exception as e:
       print('There seems to be an issue with loading the images.')
       print(e)
       return 



def create_video(images_folder, output_path, fps=10):
    """
    creates a video from a folder of images sorting on the basename of the file.

    :param
            images_folder (string): path to folder containing images
            output_path (string): path to output video
            fps (int): frames per second
        
        :return
                None
    """
    images = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f)) and (f.endswith('.png') or f.endswith('.jpg'))]
    images = sorted(images, key=lambda x: int(os.path.basename(x).split('.')[0]))

    # load first image to get dimensions
    first_image = cv2.imread(images[0])

    # if there are no images, flag error!
    if not images:
            print('Unable to find any png or jpg images in the path provided!')
            return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (first_image.shape[1], first_image.shape[0]))

    try:
        for image in images:
                frame = cv2.imread(image)
                out.write(frame)
        out.release()
        print(f'Video saved to {output_path}')

    except Exception as e:
        print('There seems to be an issue with loading the images.')
        print(e)
        return

def two_videos_side_by_side(path_video_1, path_video_2, output_path):
    """
    create a video from two videos side by side. Only works for videos of the same length.

    :param
        path_video_1 (string): path to first video
        path_video_2 (string): path to second video
        output_path (string): path to output video
    
    :return
        None
    """
    # load videos
    video_1 = cv2.VideoCapture(path_video_1)
    video_2 = cv2.VideoCapture(path_video_2)

    # get video dimensions
    width = int(video_1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_1.get(cv2.CAP_PROP_FPS))
    num_frames = int(video_1.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames_2 = int(video_2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (int(2*width), int(height)))

    if out.isOpened():
        print('Video writer is open')
    else:
        print('Video writer is not open')
        return

    try:
        for frame in range(min(num_frames, num_frames_2)):
            ret1, img1 = video_1.read()
            ret2, img2 = video_2.read()

            # concatenate images
            img = np.concatenate((img1, img2), axis=1)
            out.write(img)

        out.release()
        print(f'Video saved to {output_path}')

    except Exception as e:
        print('There seems to be an issue with loading the images.')
        print(e)
        return
    
def change_video_fps(path_to_video, output_path, fps):
    """
    Changes video fps to desired fps.

    :param
        path_to_video (string): path to video
        output_path (string): path to output video
        fps (int): frames per second
    
    :return
        None
    """
    # load video
    video = cv2.VideoCapture(path_to_video)

    # get video dimensions
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if out.isOpened():
        print('Video writer is open')
    else:
        print('Video writer is not open')
        return
    try:
        for frame in range(num_frames):
            ret, img = video.read()
            out.write(img)

        out.release()
        print(f'Video saved to {output_path}')

    except Exception as e:
        print('There seems to be an issue with loading the images.')
        print(e)
        return


def video_to_gif(video_path, output_path, start_time, end_time, fps=24, boomerang=False):
    """
    Converts a segment of a video file to a GIF.

    :param
        video_path (str): Path to the video file.
        output_path (str): Path where the GIF should be saved.
        start_time (float): Start time of the segment in seconds.
        end_time (float): End time of the segment in seconds.
        fps (int, optional): Frames per second for the output GIF. Default is 24.
        boomerang (bool, optional): If True, the GIF will go back and forth. Default is False.
    
    :return
        None
    """
    # Load the video files
    clip = VideoFileClip(video_path).subclip(start_time, end_time)

    if boomerang:
        # Reverse the video clip
        reverse_clip = clip.fx(vfx.time_mirror)
        clip = concatenate([clip, reverse_clip])
    
    # Reduce the number of frames per second
    clip = clip.set_fps(fps)
    
    # Write the GIF file
    clip.write_gif(output_path)