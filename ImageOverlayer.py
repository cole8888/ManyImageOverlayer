# Program which can overlay a huge number of images and find the average values for each of the pixels.
# When using chunk mode, make sure that you select a chunk_size small enough to not use all your RAM, this will depend on how large your images are.

# Cole Lightfoot - 26 Jan 2021 - https://github.com/cole8888/

import os
import sys
import time
import argparse
import progressbar
import numpy as np
from PIL import Image
from threading import Thread
from multiprocessing import cpu_count

# Command line arguments.
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--images",
    type=str,
    default=None,
    help='Directory of images you want to use. This should be a single level directory.',
)
parser.add_argument(
    "--width",
    type=int,
    default=512,
    help="Width to resize images to",
)
parser.add_argument(
    "--height",
    type=int,
    default=512,
    help="Height to resize images to.",
)
parser.add_argument(
    "--threads",
    type=int,
    default=cpu_count(),
    help="Number of threads to use. Automatically uses as many as your cpu has.",
)
parser.add_argument(
    "--chunk_size",
    type=int,
    default=400,
    help="Number of images to use per chunk.",
)
parser.add_argument('--autosize', action="store_true", help='Find the average width and height of all images and use this for the final image.')
parser.add_argument('--chunks', action="store_true", help='The number of chunks of size chunk_size needed to hold all images.')
args = parser.parse_args()

# Catch issues in arguments.
if(args.images is None):
    raise ValueError("You must pass an image directory path. (Single Level)")
if(args.width < 1):
    raise ValueError("Image width must be greater than 1.")
if(args.height < 1):
    raise ValueError("Image height must be greater than 1.")
if(args.chunk_size < 1):
    raise ValueError("Chunk size must be greater than 1.")
if(args.threads < 1):
    raise ValueError("Invalid number of threads. Must be greater than zero.")
if(args.threads > cpu_count()):
    if(input("\nWarning, you are going to use " + str(args.threads) + " threads. Your CPU has " + str(cpu_count()) + " threads.\nThis may cause lower performance than if you used " + str(cpu_count()) + " threads.\nDo you want to continue? (y/N): ").lower() != "y"):
        exit()

# Split up the workload into batches for each thread and handle extras.
# This is used in several different contexts.
def divide_chunks(images, n, e):
    num = 0
    chunks = []
    # Loop until n = length of images minus extras
    for i in range(0, len(images)-e, n):  
        chunks.append(images[i:i + n])
        num += 1
    
    # Append the extra images to the first chunk, to an additional one at the end if in chunked mode.
    if(e != 0):
        for i in range(len(images)-e, len(images), 1):
            chunks[0].append(images[i])
            i += 1
    return chunks

def sumImg(chunk):
    global tmp_glob_arr
    global glob_arr
    global busy
    global pbar
    global t_total_count
    # Heads up, numpy and PIL use width and height in oposite positions.
    arr = np.zeros((args.height, args.width, 3), np.float)
    # Resize the images assigned to this chunk. This is what takes up memory since we need to store the resized images.
    resizedImgs = []
    for img in chunk:
        imgF = Image.open(os.path.join(args.images, img))
        resizedImgs.append(imgF.resize((args.width, args.height)))

    # Build up average pixel intensities, casting each image as an array of floats
    index = 0
    for img in resizedImgs:
        imgArr = np.array(img, dtype=np.float)
        # Sometimes the images cannot be added for some reason, print file name to commandline.
        try:
            np.add(arr, imgArr, arr)
        except:
            # Sometimes there is an alpha channel, remove it. If this fails let user know.
            try:
                png = img.convert('RGB')
                imgArr = np.array(png, dtype=np.float)
                np.add(arr, imgArr, arr)
            except:
                print("\nUnable to average this image:", chunk[index])
        index += 1

    # Release the memeory used by the resized images. This may help prevent running out of memory.
    resizedImgs = []

    # We need to divide by the number of images in this thread's chunk in order to return the values to 0-255.
    arr = arr/len(chunk)

    # Prevent this thread from writing to glob_arr while another thread is also doing that. Loop until no longer busy.
    while(busy):
        time.sleep(0.01)
    # Not longer busy, so mark as busy and proceed.
    busy = True
    # Add this thread's work to the global array.
    np.add(tmp_glob_arr, arr, tmp_glob_arr)
    # Update the progress bar.
    t_total_count+=1
    pbar.update(t_total_count)
    busy = False
    # See if last thread and handle accordingly.
    isLast("sum")

# Sees if the thread to call this function is the last one.
def isLast(mode):
    global t_done
    global glob_arr
    global tmp_glob_arr
    global opti_size
    global size_done
    global chunk_done
    # Increment the completed threads counter.
    t_done+=1
    # If we have finished all threads, in sum mode and not in chunked mode.
    if(t_done == args.threads and mode == "sum" and not args.chunks):
        tmp_glob_arr = tmp_glob_arr/args.threads
        # Round values in array and cast as an 8-bit integer.
        glob_arr = np.array(np.round(tmp_glob_arr), dtype=np.uint8)
        # Generate and save the final image.
        out = Image.fromarray(glob_arr, mode="RGB")
        out.save("Average.png")
        print("\nSaved Average.png to the current working directory.")
    # If we have finished all threads, in sum mode and in chunked mode.
    elif(t_done == args.threads and mode == "sum" and args.chunks):
        # Reset the thread completion counter for the next chunk.
        t_done = 0
        # Divide by the number of threads to return the values to 0-255.
        tmp_glob_arr = tmp_glob_arr/args.threads
        # Add the results from this chunk to the global array.
        np.add(glob_arr, tmp_glob_arr, glob_arr)
        # Reset to prepare for next chunk.
        tmp_glob_arr = np.zeros((args.height, args.width, 3), np.float)
        # Indicate this group is done
        chunk_done = True
    # If we have finished all threads and are finding the optimal size for the final image.
    elif(t_done == args.threads and mode == "opti"):
        # Set the height and width values to the calculated optimal values
        args.width = int((opti_size[0] / args.threads))
        args.height = int((opti_size[1] / args.threads))
        # Indicate we are done finding the size, main thread can continue.
        size_done = True
        # Reset t_done for next use (for summing pixel threads).
        t_done = 0
        # We don't need this thread anymore.
        sys.exit()

# Find the optimal size for the average image. There may be multiple threads of this.
def findOptiSize(chunk):
    global opti_size
    global busy
    wSum = 0
    hSum = 0
    # For all images in this threads chunk, find the height and width of the image and add it to the running sum.
    for img in chunk:
        imgF = Image.open(os.path.join(args.images, img))
        w, h = imgF.size
        wSum += w
        hSum += h
    
    # Divide running sum by the number of images in this chunk.
    wSum /= len(chunk)
    hSum /= len(chunk)

    # Prevent this thread from writing to opti_size while another thread is also doing that. Loop until no longer busy.
    while(busy):
        time.sleep(0.01)
    busy = True
    opti_size[0] += wSum
    opti_size[1] += hSum
    busy = False
    # See if last thread and handle accordingly.
    isLast("opti")

# Get all compatible image files
files=os.listdir(args.images)
imgs=[filename for filename in files if filename[-4:] in [".png",".PNG",".jpg",".JPG",".jpeg",".JPEG"]]

# Number of images
N=len(imgs)
if(N < 1):
    raise ValueError("No images with compatible formats were found at the specified directory (Not recursive).")
print("Found " + str(N) + " images.")

# Handle edge case where we have more threads than images.
if(N<args.threads):
    args.threads = N
    # No point in using chunk mode in this case.
    args.chunks = False
    print("Warning: Number of threads is larger than the total number of images. Disabling chunk mode and setting threads to", args.threads)

# Counter for the number of threads that have finished.
t_done = 0

# Total number of threads which will run across all chunks.
t_all = 0

# Total number of threads which have completed so far across all chunks.
t_total_count = 0

# Flag to indicate when the optimal image size has been found (if autosize argument is used)
size_done = False

# Flag to indicate when the current chunk has finished.
chunk_done = False

# Flag to prevent two threads writing at once
busy = False

# Number of images per thread (except extras which are added to first thread)
elements = int(N / args.threads)

# Get 2D list of images for the threads. Each thread will have it's own list to work on.
image_chunks = divide_chunks(imgs, elements, N % args.threads)

# Image dimensions list used to transfer results from findOptiSize()
opti_size = [0, 0]

# See if we should find the optimal size for the final image.
if(args.autosize):
    print("Finding the optimal image size...")
    for i in range(args.threads):
        Thread(target=findOptiSize, args=(image_chunks[i],)).start()
    # Wait until the optimal size has been found.
    while not size_done:
        time.sleep(0.01)
    print("Average resolution is " + str(args.width) + "x" + str(args.height) + " pixels")

# Numpy array of floats to store the average values for each pixel (assume RGB images)
glob_arr = np.zeros((args.height, args.width, 3), np.float)

# Numpy array to act as a buffer before finally adding to glob_arr
tmp_glob_arr = np.zeros((args.height, args.width, 3), np.float)

# Cover edge case where batch size is greater than the number of images
if(N<args.chunk_size):
    args.chunks = False
    print("Warning: chunk_size was smaller than the total number of images. Disabling chunk mode.")

# Progress bar to show progress. We don't know how many total threads will run, so we cannot make the pbar yet.
widgets = [progressbar.Timer(), '|', progressbar.Percentage(), '|', progressbar.ETA(), ' ', progressbar.Bar()]
pbar = ""

# If chunk mode is enabled.
if(args.chunks):
    # Cover edge case where chunk_size is less than the number of threads.
    if(args.chunk_size < args.threads):
        args.chunk_size = args.threads
        print("Warning: chunk_size was smaller than the number of threads. Setting chunk_size to number of threads.")

    # How many chunks do we need to fit N images into the specified batch size
    chunks = int(N/args.chunk_size)
    image_chunks = divide_chunks(imgs, args.chunk_size, N % chunks)

    # Calculate the total number of threads that will run across all chunks. Use to make progress bar.
    t_all = len(image_chunks)*args.threads
    print(str(len(image_chunks)) + " chunks of " + str(args.chunk_size) + " images.\n")
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=t_all, term_width=100).start()

    # For all chunks
    for i in image_chunks:
        # Split up the current chunk into sub-chunks which are then assigned to threads.
        image_chunks_chunks = divide_chunks(i, int(len(i)/args.threads), len(i) % args.threads)
        # Create threads and assign them their own sub-chunk of images to look after.
        for j in range(args.threads):
            Thread(target=sumImg, args=(image_chunks_chunks[j],)).start()
        # Wait until all threads of this chunk have finished.
        while not chunk_done:
            time.sleep(0.01)
        # Reset the flag for the next chunk
        chunk_done = False
    
    # Divide by the number of chunks, this gets the values back down to 0-255.
    glob_arr = glob_arr/len(image_chunks)
    # Round values in array and cast as an 8-bit integer.
    glob_arr = np.array(np.round(glob_arr), dtype=np.uint8)
    # Generate and save the final image.
    out = Image.fromarray(glob_arr, mode="RGB")
    out.save("Average.png")
    print("Saved Average.png to the current working directory.")
else:
    print("\n")
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=args.threads, term_width=100).start()
    for i in range(args.threads):
        Thread(target=sumImg, args=(image_chunks[i],)).start()