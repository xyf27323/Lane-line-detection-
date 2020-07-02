import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


# Step 1 : Calculate camera distortion coefficients
#################################################################
def getCameraCalibrationCoefficients(chessboardname, nx, ny):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)       #(54*3)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(chessboardname)  #查找符合特定规则的文件路径名 返回一个列表
    if len(images) > 0:
        print("images num for calibration : ", len(images))
    else:
        print("No image for calibration.")
        return
    
    ret_count = 0
    for idx, fname in enumerate(images):  #idx为路径序列标号从0开始，fname为每张图的路径，enumerate枚举类型可直接将标号与列表中每个元素对应
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = (img.shape[1], img.shape[0])
        # Finde the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            ret_count += 1
            objpoints.append(objp)
            imgpoints.append(corners)
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    '''
    第一个参数objectPoints，为世界坐标系中的三维点。需要依据棋盘上单个黑白矩阵的大小，计算出（初始化）每一个内角点的世界坐标
    第二个参数imagePoints，为每一个内角点对应的图像坐标点
    第三个参数imageSize，为图像的像素尺寸大小，在计算相机的内参和畸变矩阵时需要使用到该参数
    ret返回均方根(RMS)重投影误差，通常，在良好校准中，误差应在0.1到1.0像素之间
    RMS误差为1.0意味着平均而言，这些投影点中的每个投影点都偏离其实际位置1.0 px
    
    cameraMatrix为相机的内参矩阵(3*3)
    distCoeffs为畸变矩阵,相机的5个畸变系数
    rvecs为旋转向量(每张图像都会生成属于自己的平移向量和旋转向量)
    tvecs为位移向量(每张图像都会生成属于自己的平移向量和旋转向量)
    '''
    print('Do calibration successfully')
    return ret, mtx, dist, rvecs, tvecs


nx = 9 #棋盘格横向角点数
ny = 6 #棋盘格纵向角点数
ret, mtx, dist, rvecs, tvecs = getCameraCalibrationCoefficients('camera_cal/calibration*.jpg', nx, ny)

#################################################################
# Step 2 : Undistort image
#################################################################
def undistortImage(distortImage, mtx, dist):
    return cv2.undistort(distortImage, mtx, dist, None, mtx)

# Read distorted chessboard image
#test_distort_image = cv2.imread('./camera_cal/calibration4.jpg')
#cv2.imshow("distort",test_distort_image)
#cv2.waitKey(0)

# Do undistortion
#test_undistort_image = undistortImage(test_distort_image, mtx, dist)
#cv2.imshow("undistort",test_undistort_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

##########筛选图像
#video_input = 'project_video.mp4'
#cap = cv2.VideoCapture(video_input)
#count = 1
#while(True):
#    ret, image = cap.read()
#    if ret:
#        undistort_image = undistortImage(image, mtx, dist)
#        cv2.imwrite('original_image/' + str(count) + '.jpg', undistort_image)
#        count += 1
#    else:
#        break
#cap.release()

#################################################################
# Step 3 : Warp image based on src_points and dst_points
#################################################################
# The type of src_points & dst_points should be like
# np.float32([ [0,0], [100,200], [200, 300], [300,400]])
def warpImage(image, src_points, dst_points):
    image_size = (image.shape[1], image.shape[0])
    # rows = img.shape[0] 720
    # cols = img.shape[1] 1280
    M = cv2.getPerspectiveTransform(src, dst)#通过变换前后的点的位置求得透视变换矩阵，投射变换至少需要四组变换前后对应的点坐标
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(image, M,image_size, flags=cv2.INTER_LINEAR)
    
    return warped_image, M, Minv

#test_distort_image = cv2.imread('original_image/16.jpg')

# 畸变修正
#test_undistort_image = undistortImage(test_distort_image, mtx, dist)
#cv2.imshow("undistort",test_undistort_image)
#cv2.waitKey(0)
# 左图梯形区域的四个端点
src = np.float32([[580, 460], [700, 460], [1096, 720], [200, 720]])
# 右图矩形区域的四个端点
dst = np.float32([[300, 0], [950, 0], [950, 720], [300, 720]])

#test_warp_image, M, Minv = warpImage(test_undistort_image, src, dst)
#cv2.imshow("warp",test_warp_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#将RGB通道的图，转为HLS通道的图，随后对L通道进行分割处理，提取图像中白色的车道线
def hlsLSelect(img, thresh=(220, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]   #0H 1L 2S
    l_channel = l_channel*(255/np.max(l_channel))
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
    return binary_output

#将RGB通道的图，转为Lab通道的图，随后对b通道进行分割处理，提取图像中黄色的车道线
def labBSelect(img, thresh=(195, 255)):
    # 1) Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_b = lab[:,:,2]
    # don't normalize if there are no yellows in the image
    if np.max(lab_b) > 170:#原博给的100 但遇全白车道线有问题 调整为170
        lab_b = lab_b*(255/np.max(lab_b))
    # 2) Apply a threshold to the L channel
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
    # 3) Return a binary image of threshold result
    return binary_output  #单通道

#将两个通道分割的图像合并
#hlsL_binary = hlsLSelect(test_warp_image)
#labB_binary = labBSelect(test_warp_image)
#combined_binary = np.zeros_like(hlsL_binary)
#combined_binary[(hlsL_binary == 1) | (labB_binary == 1)] = 1

#cv2.imshow("combined_binary",labB_binary)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#使用直方图找左右车道线大致位置，其中leftx_base和rightx_base即为左右车道线所在列的大致位置
## Take a histogram of the bottom half of the image
#histogram = np.sum(combined_binary[combined_binary.shape[0]//2:,:], axis=0) #shape[0]=720  sum(axis=0)即在第0维（对每行）做加操作，共1280列，得到1280个值
## Create an output image to draw on and visualize the result
#out_img = np.dstack((combined_binary, combined_binary, combined_binary))  #按深度即通道数叠加
## Find the peak of the left and right halves of the histogram
## These will be the starting point for the left and right lines
#midpoint = np.int(histogram.shape[0]//2)    #图像列的中点 1280//2
#leftx_base = np.argmax(histogram[:midpoint])  #np.argmax  取出目标数组中元素最大值所对应的索引
#rightx_base = np.argmax(histogram[midpoint:]) + midpoint

#cv2.imshow("histogram",out_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#################################################################
# Step 5 : Detect lane lines through moving window
#################################################################
def find_lane_pixels(binary_warped, nwindows, margin, minpix):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()  #返回非零元素的位置
    nonzeroy = np.array(nonzero[0]) #非零元素所在的行 即纵坐标
    nonzerox = np.array(nonzero[1])  #非零元素所在的列 即横坐标
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin    #每个搜索框宽200（2margin）
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]   #返回xy都满足约束的索引位置，即方框中所有车道线（非0）像素
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds) #[[1 2 3],[4 5],...[6 7 8 9]] nwindows维
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels阈值, recenter next window on their mean position
        if len(good_left_inds) > minpix:             #大于一定阈值认为是一段车道线
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))   #重新计算方框横坐标 否则不改变横坐标继续向上移动方框
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)  #列表转换为1维数组 [1 2 3 4 5 6 7 8 9]
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]     #滑动窗找到的所有左车道线横坐标
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(
        binary_warped, nwindows, margin, minpix)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)   #函数np.polyfit(x横,y纵,n次)前边是x 后边是y
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )  #均匀产生纵向点(0起,719止,720个)
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]   #blue
    out_img[righty, rightx] = [0, 0, 255]  #red

    # Plots the left and right polynomials on the lane lines
#    plt.plot(left_fitx, ploty, color='yellow')
#    plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit, ploty



#################################################################
# Step 6 : Track lane lines based the latest lane line result
#################################################################
def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def search_around_poly(binary_warped, left_fit, right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 60

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit, right_fit = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    
    return result, left_fit, right_fit, ploty




#################################################################
# Step 8 : Draw lane line result on undistorted image
#################################################################
def drawing(undist, bin_warped, color_warp, left_fit, right_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result



#最终图片测试

#out_img, left_fit, right_fit, ploty = fit_polynomial(combined_binary, nwindows=9, margin=80, minpix=40)
#testresult = drawing(test_undistort_image, combined_binary, test_warp_image, left_fit, right_fit)
#cv2.imshow("testresult",testresult)
#cv2.imshow("combined_binary",combined_binary)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#最终视频测试

video_input = 'project_video.mp4'
video_output = 'result_video.mp4'

cap = cv2.VideoCapture(video_input)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_output, fourcc, 20.0, (1280, 720))

detected = False

while(True):
    ret, image = cap.read()
    if ret:
        undistort_image = undistortImage(image, mtx, dist)
        warp_image, M, Minv = warpImage(undistort_image, src, dst)
        hlsL_binary = hlsLSelect(warp_image)
#        labB_binary = labBSelect(warp_image, (205, 255))
        labB_binary = labBSelect(warp_image)
        combined_binary = np.zeros_like(hlsL_binary)
        combined_binary[(hlsL_binary == 1) | (labB_binary == 1)] = 1

        if detected == False:  #只有第一帧用滑动窗搜索 后面帧都在上次曲线附近搜索
            left_fit = []
            right_fit = []
            ploty = []
            out_img, left_fit, right_fit, ploty = fit_polynomial(combined_binary, nwindows=9, margin=80, minpix=40)
            if (len(left_fit) > 0) & (len(right_fit) > 0) :
                detected = True
            else :
                detected = False
        else:
            track_result, left_fit, right_fit, ploty,  = search_around_poly(combined_binary, left_fit, right_fit) #left_fitx?
            if (len(left_fit) > 0) & (len(right_fit) > 0) :#第一次进这个函数leftfit参数为空没传进来
                detected = True
            else :
                detected = False
        
        result = drawing(undistort_image, combined_binary, warp_image, left_fit, right_fit)
        
        out.write(result)
    else:
        break
print('Do video_output successfully')      
cap.release()
out.release()










