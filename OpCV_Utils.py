######################################################################################################################################
##################### OpenCV Utilities Module by Matheus D. Pereira (https://github.com/Mafeuz) ######################################
######################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2

######################################################################################################################################
# FUNCTION LIST:
# imgFourierTransform(img)
# invImgFourierTransform(fft)
# img_translation(img, dx, dy)
# img_rotation(img, angle, pivot, keep_full_img=False)
# affine_transform(img, pts1, pts2)
# gray_histogram(img, normalize=True)
# color_histogram(img, normalize=True)
# show_single_img_plt(img, title, fig_size=(15,15), show_axis=False)
# show_multiple_imgs_plt(images_array, titles_array, fig_size = (15,15), show_axis=False)
# stackImgs(images_array, sep_lines=False, scale=0.5)
# color_filtering(img, boundaries)
# thresh_img(img, thresh_type='Binary Thresh', thresh=230, block_size=41, C=8, return_gray=False)
# custom_canny(img, blur_kernel_size = (5,5), kernel_size = (3,3), canny_thresh = (100,100), order = 1, dil_level = 0, ero_level = 0)
# canny_trackbars(img, img_resize=(600,500), krnl_size = (3,3), stackImgs=False, stack_scale=(0.5))
# find_contours(img, c_thresh = (100,100), dil = 1, ero = 0)
# reorder4points(points)
# img_warping_ref_obj(img, ref_points, ref_obj_W, ref_obj_H, pad=0)
# img_homography(img, points1, points2, pad=0)
# pointMouseCallback(event, x, y, flags, param)
# selectPolygonMouseCallback(event, x, y, flags, param)
# drawPolygon(img, p1, p2, p3, p4)
# drawBoundingBox(img, x, y, w, h, text, color)
# contrast_CLAHE(img, clipLimit=3.0, tileGridSize=(8,8))
# img_upscale_3x(img, path)
# img_upscale_8x(img, path)

######################################################################################################################################
######################################################################################################################################
def imgFourierTransform(img):
    
    # Convert img to gray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute the 2D-FFT:
    fft = np.fft.fft2(img_gray)
    
    # Get the phase spectrum:
    phase_spectrum = np.angle(fft)
    
    # Shift adjust low frequencies to the center (phase info is lost) and get magnitude:
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = 20*np.log(np.abs(fft_shifted))
    
    # Adjust imgs to output:
    magnitude_spectrum = magnitude_spectrum.astype('uint8')
    magnitude_spectrum = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
    
    phase_spectrum = phase_spectrum.astype('uint8')
    phase_spectrum = cv2.cvtColor(phase_spectrum, cv2.COLOR_GRAY2RGB)
    
    return fft, magnitude_spectrum, phase_spectrum

######################################################################################################################################
def invImgFourierTransform(fft):
    # Recover img from the fft using the inverse fft:
    img_recovered = np.abs(np.fft.ifft2(fft))

    # Convert to show:
    img_recovered = img_recovered.astype('uint8')
    img_recovered = cv2.cvtColor(img_recovered, cv2.COLOR_GRAY2RGB)
    
    return img_recovered

######################################################################################################################################
def img_translation(img, dx, dy):
    M = np.float32([[1, 0, dx],
                    [0, 1, dy]])
    
    shifted = cv2.warpAffine(img, M, (0,0))
    
    return shifted

######################################################################################################################################
def img_rotation(img, angle, pivot, keep_full_img=False):
    
    (h, w) = img.shape[:2]
    (cX, cY) = pivot
    
    if keep_full_img:
        
        # Our img might have a extreme point as [0,0,0], that will cause a bad border identification.
        # So its needed to change it:
        if (angle !=0):
            img[np.where((img == [0,0,0]).all(axis = 2))] = [1, 0, 0]
        
        # Create space for keeping full img:
        bboard = np.zeros_like(img)
        maxdim = np.argmax([h, w])
        dims = (np.max([2*h, 2*w]), np.max([2*h, 2*w]))
        bboard = cv2.resize(bboard, (dims))
               
        # Put original img in the center of the black board:
        bboard[bboard.shape[0]//4:(bboard.shape[0]//4 + img.shape[0]),
               bboard.shape[0]//4:(bboard.shape[0]//4 + img.shape[1]),:] = img
                  
        # New img = black board:
        img = bboard
                                
        # Rotate new img:
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                
        # Get img region extremes:
        loc = np.where((rotated != [0,0,0]).all(axis = 2))
        extTop = np.min(loc[0])
        extLeft = np.min(loc[1])
        
        # Move it to desired region to (0,0)
        M = np.float32([[1, 0, -extLeft],
                        [0, 1, -extTop]])
        
        rotated = cv2.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]))
        
        # Cut not desired region:
        loc = np.where((rotated != [0,0,0]).all(axis = 2))
        extRight = np.max(loc[0])
        extBot = np.max(loc[1])
        
        rotated = rotated[:extRight,:extBot,:]
            
    else:
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
    
    return rotated

######################################################################################################################################
def affine_transform(img, pts1, pts2):
    h, w, _ = img.shape

    M = cv2.getAffineTransform(pts1, pts2)
    transformed = cv2.warpAffine(img, M, (w, h))
    
    return transformed

######################################################################################################################################
def gray_histogram(img, normalize=True):
    # Gray-scale histogram:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    
    # Plot:
    plt.subplot()
    plt.title('Gray-Scale Histogram');
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    plt.xlim([0, 256])

    if normalize:
        # Normalizing:
        hist /= hist.sum()
        plt.ylabel('% of Pixels')
        
    plt.plot(hist)
   
    return hist

######################################################################################################################################
def color_histogram(img, normalize=True):
    # expecting BGR img
    # Split Channels:
    chans = cv2.split(img)
    colors = ('b', 'g', 'r')
    
    # Plot
    plt.figure()
    plt.title('Flatten Color Histogram');
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')
    plt.xlim([0, 256])

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        
        # Normalizing:
        if normalize:
            hist /= hist.sum()
            plt.ylabel('% of Pixels')
            
        plt.plot(hist, color=color)
        
    return hist

######################################################################################################################################
def show_single_img_plt(img, title, fig_size=(15,15), show_axis=False):
    fig, axis = plt.subplots(figsize = fig_size)
  
    axis.imshow(img)
    axis.set_title(title, fontdict = {'fontsize': 22, 'fontweight': 'medium'})
    
    if not show_axis:
        axis.axis('off')
        
######################################################################################################################################
def show_multiple_imgs_plt(images_array, titles_array, fig_size = (15,15), show_axis=False):
    # Function for outputing plt subplots from images (RGB).
    # Each row of images must have the same number of elements as the others.
    # list/array form: [row1,row2,...rowN], row = [element1, element2,...elementN]
    
    if (len(images_array) > 1) & (len(images_array[0]) > 1):
        fig, axis = plt.subplots(len(images_array), len(images_array[0]), figsize = fig_size)

        for i in range(len(images_array)):
            for j in range(len(images_array[0])):
                axis[i][j].imshow(images_array[i][j])
                axis[i][j].set_anchor('NW')
                axis[i][j].set_title('{}'.format(titles_array[i][j]), fontdict = {'fontsize': 15, 'fontweight': 'medium'}, pad = 10)
                if not show_axis:
                    axis[i][j].axis('off')

    if (len(images_array) == 1):
        fig, axis = plt.subplots(1, len(images_array[0]), figsize = fig_size)
        for j in range(len(images_array[0])):
            axis[j].imshow(images_array[0][j])
            axis[j].set_anchor('NW')
            axis[j].set_title('{}'.format(titles_array[0][j]), fontdict = {'fontsize': 15, 'fontweight': 'medium'}, pad = 10)
            if not show_axis:
                axis[j].axis('off')

    if (len(images_array[0]) == 1):
        fig, axis = plt.subplots(len(images_array), 1, figsize = fig_size)
        for j in range(len(images_array)):
            axis[j].imshow(images_array[j][0])
            axis[j].set_anchor('NW')
            axis[j].set_title('{}'.format(titles_array[j][0]), fontdict = {'fontsize': 15, 'fontweight': 'medium'}, pad = 10)
            if not show_axis:
                axis[j].axis('off')
            
######################################################################################################################################
def stackImgs(images_array, sep_lines=False, scale=0.5):
    # Function for rescaling and stacking cv2 BGR images together.
    # list/array form: [row1,row2,...rowN], row = [element1, element2,...elementN]
        
    ##################################################################################################################
    # Resize images based on the shape of the first image:
    for i in range(len(images_array)):
        for j in range(len(images_array[i])):
            images_array[i][j] = cv2.resize(images_array[i][j], (images_array[0][0].shape[1], images_array[0][0].shape[0]))
    
    # Images Rescaling and Convert Gray-scale to BGR if needed:
    for i in range(len(images_array)):
        for j in range(len(images_array[i])):
            images_array[i][j] = cv2.resize(images_array[i][j], (0, 0), None, scale, scale)
            
            if (len(images_array[i][j].shape) == 2):
                images_array[i][j] = cv2.cvtColor(images_array[i][j], cv2.COLOR_GRAY2BGR)
                
            if (len(images_array[i][j].shape) == 3):
                if (images_array[i][j].shape[2] == 1):
                    images_array[i][j] = cv2.cvtColor(images_array[i][j], cv2.COLOR_GRAY2BGR)
    
    ##################################################################################################################
    # Let's equalize rows number of images:
    lens = [1]*len(images_array)
    
    for i in range(len(images_array)):
        lens[i] = len(images_array[i])
    
    # get max_len to add a blank image if necessary:
    max_len = np.max(lens)
    
    for i in range(len(images_array)):
        if (len(images_array[i]) < max_len):
            blank_image = np.ones((images_array[0][0].shape[0], images_array[0][0].shape[1] , 3),dtype=np.uint8)*150
        
            for j in range(max_len - len(images_array[i])):
                images_array[i].append(blank_image)
                
    ##############################################################################################################
    # Stacking images:
    for i in range(len(images_array)):
        for j in range(len(images_array[i])):
            if (j == 0):
                h_stack = images_array[i][j]
            if (j > 0):
                h_stack = np.hstack([h_stack, images_array[i][j]])
        
        if (i == 0):                          
            v_stack = h_stack
        if (i > 0):                          
            v_stack = np.vstack([v_stack, h_stack])
                      
    ##############################################################################################################
    # Paint Separation Lines:
    if sep_lines:
        
        # Horizontal Lines:
        if (len(images_array) > 0):
            for i in range(len(images_array)-1):
                cv2.line(v_stack, (0, (i+1)*(v_stack.shape[0]//len(images_array))), 
                        (v_stack.shape[1], (i+1)*(v_stack.shape[0]//len(images_array))), (255, 0, 0), 2)
                
        # Vertical Lines:
        for i in range(max_len-1):
            cv2.line(v_stack, ((i+1)*(v_stack.shape[1]//max_len), 0), 
                     ((i+1)*(v_stack.shape[1]//max_len), v_stack.shape[0]), (255, 0, 0), 2)
            
    ##############################################################################################################
                                  
    return v_stack

######################################################################################################################################
def color_filtering(img, boundaries):

    # loop over the boundaries
    for (lower, upper) in boundaries:

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8") # Lower color limit
        upper = np.array(upper, dtype = "uint8") # Upper color limit

        mask = cv2.inRange(img, lower, upper) # mask wit in range of lower to upper
        output = cv2.bitwise_and(img, img, mask = mask)
         
    return output

######################################################################################################################################
def thresh_img(img, thresh_type='Binary Thresh', thresh=230, block_size=41, C=8, return_gray=False):
    
    if (thresh_type == 'Binary Thresh'):
        _, img_thresh = cv2.threshold(img.copy(), thresh, 255, cv2.THRESH_BINARY)
        
    if (thresh_type == 'Otsu'):
        _, img_thresh = cv2.threshold(img.copy()[:,:,0], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if (thresh_type == 'Adaptive Mean C'):
        img_thresh = cv2.adaptiveThreshold(img.copy()[:,:,0], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
        
    if (thresh_type == 'Adaptive Gaussian C'):
        img_thresh = cv2.adaptiveThreshold(img.copy()[:,:,0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

    if not return_gray:
        img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
        return img_thresh
    
    return img_thresh
    
######################################################################################################################################
def custom_canny(img, blur_kernel_size = (5,5), kernel_size = (3,3), canny_thresh = (100,100), order = 1, dil_level = 0, ero_level = 0):
    # Canny Processing:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, blur_kernel_size, 1)
    canny_img = cv2.Canny(blur_img, canny_thresh[0], canny_thresh[1])
    
    kernel = np.ones(kernel_size)
    
    if (order == 1):
        img_dilation = cv2.dilate(canny_img, kernel, iterations = dil_level)
        img_thresh = cv2.erode(img_dilation, kernel, iterations = ero_level)
        
    if (order == -1):
        img_erosion = cv2.erode(canny_img, kernel, iterations = ero_level)
        img_thresh = cv2.dilate(img_erosion, kernel, iterations = dil_level)
    
    return img_thresh

######################################################################################################################################
def canny_trackbars(img, img_resize=(600,500), krnl_size = (3,3), stack_imgs=False, stack_scale=(0.5)):
    
    print('Press K to break.\n')
    
    ############################################################################
    # trackbar callback function does nothing but required for trackbar
    def nothing(x):
        pass

    # Canny Controls Trackbar:
    cv2.namedWindow('Canny Controls')
    cv2.createTrackbar('Dilation','Canny Controls', 0,10, nothing)
    cv2.createTrackbar('Erosion','Canny Controls', 0,10, nothing)
    cv2.createTrackbar('Canny Thresh X','Canny Controls', 100,255, nothing)
    cv2.createTrackbar('Canny Thresh Y','Canny Controls', 100,255, nothing)
    ############################################################################

    img = cv2.resize(img, img_resize)
        
    while True:

        # Keyboard Controls:

        key = cv2.waitKey(1) or 0xff   

        if key == ord('k'):
            break

        ############################################################################

        dilation = int(cv2.getTrackbarPos('Dilation','Canny Controls'))
        erosion = int(cv2.getTrackbarPos('Erosion','Canny Controls'))
        cny_thresh = (int(cv2.getTrackbarPos('Canny Thresh X','Canny Controls')), 
                      int(cv2.getTrackbarPos('Canny Thresh Y','Canny Controls')))

        img_canny = custom_canny(img.copy(), kernel_size = krnl_size, canny_thresh = cny_thresh, 
                                   dil_level = dilation, ero_level = erosion)

        if stack_imgs:
            img_canny = stackImgs([[img, img_canny]], scale = stack_scale)

        ############################################################################

        cv2.imshow('Canny Controls', img_canny)

    cv2.destroyAllWindows()

    print('=============================================')
    print('Final Dilation Iterations:', dilation)
    print('Final Erosion Iterations:', erosion)
    print('Final Canny Thresh:', cny_thresh)
    print('=============================================')
    
    return img_canny

######################################################################################################################################
def find_contours(img, c_thresh = (100,100), dil = 1, ero = 0):
    
    img_canny = custom_canny(img.copy(), blur_kernel_size = (5,5), kernel_size = (3,3), 
                             canny_thresh = c_thresh, order = 1, dil_level = dil, ero_level = ero)

    contours, hiearchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    conts = []

    for i, c in enumerate(contours):
        
            # Get contour Area:
            area = cv2.contourArea(c)
            
            # Get contour perimiter (closed):
            perimeter = cv2.arcLength(c, True)
                
            # Get contour approx corner points:
            approxCorners = cv2.approxPolyDP(c, 0.02*perimeter, True)
            
            # Centroid:
            M = cv2.moments(c)

            # calculate x,y coordinate of centroid
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Get contour extreme points:
                extLeft = c[c[:, :, 0].argmin()][0]
                extRight = c[c[:, :, 0].argmax()][0]
                extTop = c[c[:, :, 1].argmin()][0]
                extBot = c[c[:, :, 1].argmax()][0]

                bbox_start_point = (extLeft[0], extTop[1])
                bbox_end_point = (extRight[0], extBot[1])

            else:
                cX, cY = None, None
                bbox_start_point = None
                bbox_end_point = None

            conts.append([c, area, (cX, cY), bbox_start_point, bbox_end_point, perimeter, approxCorners])

    # Sort contours by area size (biggest to smaller)
    conts = sorted(conts, key = lambda x:x[1], reverse = True)
    
    return conts, img_canny

######################################################################################################################################
def reorder4points(points):
    # redorder 4 points ref to their coordinates
    # points format = np.array([[[p1x, p1y]], [[p2x, p2y]], [[p3x, p3y]], [[p4x, p4y]]])
    reordered_points = np.zeros_like(points)
    points = points.reshape((4,2))
    
    add = points.sum(1)
    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    
    diff = np.diff(points, axis = 1)
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]
    
    return reordered_points

######################################################################################################################################
def img_warping_ref_obj(img, ref_points, ref_obj_W, ref_obj_H, pad=0):
    # ref_points format = np.array([[[p1x, p1y]], [[p2x, p2y]], [[p3x, p3y]], [[p4x, p4y]]])
    
    ###################################################
    # Reordering points if needed:
    reordered_points = np.zeros_like(ref_points)
    points = ref_points.reshape((4,2))
    
    add = points.sum(1)
    reordered_points[0] = points[np.argmin(add)]
    reordered_points[3] = points[np.argmax(add)]
    
    diff = np.diff(points, axis = 1)
    reordered_points[1] = points[np.argmin(diff)]
    reordered_points[2] = points[np.argmax(diff)]
    
    points = reordered_points
    ###################################################
    
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[ref_obj_W,0],[0,ref_obj_H],[ref_obj_W,ref_obj_H]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    img_warped = cv2.warpPerspective(img, matrix, (ref_obj_W, ref_obj_H))
    img_warped = img_warped[pad:img_warped.shape[0]-pad, pad:img_warped.shape[1]-pad]
    
    return img_warped

######################################################################################################################################
def img_homography(img, points1, points2, pad=0):
    # Points must have to be defined in the same resolution for the source and destination images.
    # points format = np.array([[[p1x, p1y]], [[p2x, p2y]], [[p3x, p3y]], [[p4x, p4y]]])
    
    ###################################################
    reordered_points1 = np.zeros_like(points1)
    points1 = points1.reshape((4,2))
    
    add = points1.sum(1)
    reordered_points1[0] = points1[np.argmin(add)]
    reordered_points1[3] = points1[np.argmax(add)]
    
    diff = np.diff(points1, axis = 1)
    reordered_points1[1] = points1[np.argmin(diff)]
    reordered_points1[2] = points1[np.argmax(diff)]
    
    points1 = reordered_points1
    ###################################################
    reordered_points2 = np.zeros_like(points2)
    points2 = points2.reshape((4,2))
    
    add = points2.sum(1)
    reordered_points2[0] = points2[np.argmin(add)]
    reordered_points2[3] = points2[np.argmax(add)]
    
    diff = np.diff(points2, axis = 1)
    reordered_points2[1] = points2[np.argmin(diff)]
    reordered_points2[2] = points2[np.argmax(diff)]
    
    points2 = reordered_points2
    ###################################################
    
    pts1 = np.float32(points1)
    pts2 = np.float32(points2)
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
 
    W = img.shape[1]
    H = img.shape[0]
    img_warped = cv2.warpPerspective(img, matrix, (W,H))
    img_warped = img_warped[pad:img_warped.shape[0]-pad, pad:img_warped.shape[1]-pad]
    
    return img_warped

######################################################################################################################################
def pointMouseCallback(event, x, y, flags, param):
    # Set window and mouse callback:
    # cv2.namedWindow('Video Stream')
    # cv2.setMouseCallback('Video Stream', pointMouseCallback)
    # define globals: mouse = (0,0), p = (0,0), clicked = False
    
    global mouse, p, clicked
    
    mouse = (x, y)
        
    if event == cv2.EVENT_LBUTTONDOWN:   
        clicked = True
        p = (x, y)
    
    if clicked == True:
        if event == cv2.EVENT_MOUSEMOVE:
            p = (x, y)
              
        if event == cv2.EVENT_LBUTTONUP:
            clicked = False

######################################################################################################################################
def selectPolygonMouseCallback(event, x, y, flags, param):
    
    # Global Aux Variables Should Start As:
    # clicked = False
    # move_points = [False, False, False, False]
    
    # Starting Points Example:
    # p1 = (image.shape[1]//3, image.shape[0]//4)
    # p2 = (2*image.shape[1]//3, image.shape[0]//4)
    # p3 = (image.shape[1]//3, 3*image.shape[0]//4)
    # p4 = (2*image.shape[1]//3, 3*image.shape[0]//4)
    
    # How to set callback to a window:
    # cv2.namedWindow('Image')
    # cv2.setMouseCallback('Image', selectPolygonMouseCallback)

    global p1, p2, p3, p4, clicked, move_points
    
    p_radius = 10
       
    if event == cv2.EVENT_LBUTTONDOWN:
        
        if (((p1[0] - p_radius) < x < (p1[0] + p_radius)) & ((p1[1] - p_radius) < y < (p1[1] + p_radius))):
            p1 = (x, y)
            clicked = True
            move_points = [True, False, False, False]
            
        elif (((p2[0] - p_radius) < x < (p2[0] + p_radius)) & ((p2[1] - p_radius) < y < (p2[1] + p_radius))):
            p2 = (x, y)
            clicked = True
            move_points = [False, True, False, False]
            
        elif (((p3[0] - p_radius) < x < (p3[0] + p_radius)) & ((p3[1] - p_radius) < y < (p3[1] + p_radius))):
            p3 = (x, y)
            clicked = True
            move_points_dest = [False, False, True, False]
            
        elif (((p4[0] - p_radius) < x < (p4[0] + p_radius)) & ((p4[1] - p_radius) < y < (p4[1] + p_radius))):
            p4 = (x, y)
            clicked = True
            move_points = [False, False, False, True]
    
    if clicked == True:
        if event == cv2.EVENT_MOUSEMOVE:
            if (move_points[0] == True):
                p1 = (x, y)
            if (move_points[1] == True):
                p2 = (x, y)
            if (move_points[2] == True):
                p3 = (x, y)
            if (move_points[3] == True):
                p4 = (x, y)
            
        if event == cv2.EVENT_LBUTTONUP:
            clicked = False
            move_points = [False, False, False, False]

######################################################################################################################################
def drawPolygon(img, p1, p2, p3, p4):
    
    p_outer_radius = 10
    p_inner_radius = 2
    
    cv2.circle(img, p1, p_outer_radius, (255,0,0), thickness = 1, lineType = cv2.LINE_AA)
    cv2.circle(img, p2, p_outer_radius, (255,0,0), thickness = 1, lineType = cv2.LINE_AA)
    cv2.circle(img, p3, p_outer_radius, (255,0,0), thickness = 1, lineType = cv2.LINE_AA)
    cv2.circle(img, p4, p_outer_radius, (255,0,0), thickness = 1, lineType = cv2.LINE_AA)

    cv2.line(img, p1, p2, (255, 0, 0), 3)
    cv2.line(img, p2, p4, (255, 0, 0), 3)
    cv2.line(img, p1, p3, (255, 0, 0), 3)
    cv2.line(img, p3, p4, (255, 0, 0), 3)
    
    cv2.circle(img, p1, p_inner_radius, (0,0,255), thickness = -1, lineType = cv2.LINE_AA)
    cv2.circle(img, p2, p_inner_radius, (0,0,255), thickness = -1, lineType = cv2.LINE_AA)
    cv2.circle(img, p3, p_inner_radius, (0,0,255), thickness = -1, lineType = cv2.LINE_AA)
    cv2.circle(img, p4, p_inner_radius, (0,0,255), thickness = -1, lineType = cv2.LINE_AA)

######################################################################################################################################
def drawBoundingBox(img, x, y, w, h, text, color):
    
    # add litle background to text info:
    backg = np.full((img.shape), (0,0,0), dtype=np.uint8)
    cv2.putText(backg, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    fx,fy,fw,fh = cv2.boundingRect(backg[:,:,2])
                
    # Draw obj bbox:
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2) 
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, -1) 
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, 3) 
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
    
    return img

######################################################################################################################################
def contrast_CLAHE(img, clipLimit=3.0, tileGridSize=(8,8)):
    
    # Contrast Limited Adaptaive Histogram Equalization
    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    L, A, B = cv2.split(img)
    CLAHE = cv2.createCLAHE(clipLimit, tileGridSize)
    CL = CLAHE.apply(L)
    merge = cv2.merge((CL, A, B))
    output = cv2.cvtColor(merge, cv2.COLOR_LAB2BGR)
    
    return output 

######################################################################################################################################
def img_upscale_3x(img, path):

    # path = 'FSRCNN_x3.pb'

    sr3x = cv2.dnn_superres.DnnSuperResImpl_create()
    sr3x.readModel(path3x)
    sr3x.setModel("fsrcnn", 3)
    img_upscale = sr3x.upsample(img)
    
    return img_upscale

######################################################################################################################################
def img_upscale_8x(img, path):
    
    # path = 'LapSRN_x8.pb'
    
    sr8x = cv2.dnn_superres.DnnSuperResImpl_create()
    sr8x.readModel(path)
    sr8x.setModel('lapsrn', 8)
    img_upscale = sr8x.upsample(img)
    
    return img_upscale

######################################################################################################################################