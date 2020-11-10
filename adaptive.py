import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

def guided_filter(src,p,r,eps):
    #print('src',src)
    #src = np.ones((722,480))
    h,w = src.shape[0],src.shape[1]

    N = colfilt_boxfilter(np.ones((h,w)),r)

    l_mean = colfilt_boxfilter(src,r) / N
    p_mean = colfilt_boxfilter(p,r) / N
    lp_mean = colfilt_boxfilter(src * p,r) / N
    lp_cov = lp_mean - l_mean * p_mean

    ll_mean = colfilt_boxfilter(src * src,r) / N
    l_var = ll_mean - l_mean * l_mean

    a = lp_cov / (l_var + eps)#eq9
    b = p_mean - a * l_mean #eq10

    a_mean = colfilt_boxfilter(a,r) / N
    b_mean = colfilt_boxfilter(b,r) / N


    q = a_mean * src + b_mean #eq7
    return q

def guided_filter2(src,p,r,eps):
    #print('src',src)
    #src = np.ones((722,480))
    h,w = src.shape[0],src.shape[1]

    N = colfilt_boxfilter(np.ones((h,w)),r)

    l_mean = colfilt_boxfilter(src,r) / N
    #p_mean = colfilt_boxfilter(src,r) / N
    lp_mean = colfilt_boxfilter(src * src,r) / N
    lp_cov = lp_mean - l_mean * l_mean

    #ll_mean = colfilt_boxfilter(src * src,r) / N
    #l_var = lp_mean - l_mean * l_mean

    a = lp_cov / (lp_cov + eps)#eq9
    b = l_mean - a * l_mean #eq10

    a_mean = colfilt_boxfilter(a,r) / N
    b_mean = colfilt_boxfilter(b,r) / N


    q = a_mean * src + b_mean #eq7
    return q


def colfilt_boxfilter(img,r):
    h , w = img.shape[0],img.shape[1]
    imDst = np.zeros((h,w))
    #累计y
    imCum = np.cumsum(img,axis=0)
    imDst[0 : r + 1,:] =imCum[r : 2 * r + 1,:]# 不算右边
    imDst[r + 1:h - r,:] = imCum[2 * r + 1:h,:] - imCum[0:h - 2 * r - 1,:]
    #最后一行 复制 r遍 ,减去 [0 + (h-2r) ,0 + h - r]
    imDst[h - r:h,:] = np.tile(imCum[h - 1,:],(r,1)) - imCum[h - 2 * r - 1:h - r - 1,:]

    imCum = np.cumsum(imDst, axis=1)

    imDst[:,0: r + 1] = imCum[:,r: 2 * r + 1]  # 不算右边
    imDst[:,r + 1:w - r] = imCum[:,2 * r + 1:w] - imCum[:,0:w - 2 * r - 1]
    #print(imCum[:,w - 1])
    #print(imCum[h - 1,:])
    # 最后一行 复制 r遍 ,减去 [0 + (h-2r) ,0 + h - r]
    #imDst[:,w - r:w] = np.tile(imCum[:,w - 1:w], (1,r)) - imCum[:,w - 2 * r - 1:w - r - 1]

    #print(imDst)
    return imDst



def maxfilter(img,kernel_size):

    filer = np.ones((kernel_size,kernel_size))
    bias = (kernel_size - 1) // 2
    h_old,w_old = img.shape[0],img.shape[1]
    out = np.zeros((h_old, w_old))
    insert_col = np.zeros((h_old,bias))


    img = np.insert(img, [0], insert_col, axis=1)
    img = np.insert(img, [img.shape[1]], insert_col, axis=1)
    print('img 1',img.shape)
    w_new = img.shape[1]
    insert_row = np.zeros((bias , w_new))
    img = np.insert(img, [0], insert_row, axis=0)
    img = np.insert(img, [img.shape[0]], insert_row, axis=0)
    print('filter',img.shape)


    for x in range(0,img.shape[1] - kernel_size+ 1):
        for y in range(0,img.shape[0] - kernel_size + 1):
            tmp = img[y : y + 2 * bias + 1, x : x + 2 * bias + 1] * filer
            max_v = np.max(tmp)
            out[y,x] = max_v

    #print(out)
    return out


def SimplestColorBalance(im_orig, satLevel1, satlevel2, num):
    print(im_orig.shape)
    d  = 0
    if np.ndim(im_orig) == 3:
        h,w,c = im_orig.shape[0],im_orig.shape[1],im_orig.shape[2]
        d = c
        imRGB_orig = np.zeros((c,h * w))
        for i in range(c):
            imRGB_orig[i,:] = np.reshape(im_orig[:,:,i],(1,h * w))
    else :
        h,w = im_orig.shape[0],im_orig.shape[1]
        c = 1
        d = c
        imRGB_orig = np.reshape(im_orig,(1, h * w))
    #灰度图没写

    q = [satLevel1,1 - satlevel2]
    print('d :',d)
    imRGB = np.zeros((d,h * w))

    for ch in range(d):
        #找两个位置的分位数，猜
        print(imRGB_orig[ch,:])
        t = imRGB_orig[ch,:]
        s = t[~np.isnan(t)]
        print(s)
        tiles = np.quantile(s, q)
        #tiles = np.quantile(imRGB_orig[ch,:],q)
        temp = imRGB_orig[ch,:]
        #把数值 控制在两个分位数之间
        temp[temp < tiles[0]] = tiles[0]
        temp[temp > tiles[1]] = tiles[1]
        imRGB[ch,:] = temp
        bottom = np.min(imRGB[~np.isnan(imRGB)])
        top = np.max(imRGB[~np.isnan(imRGB)])
        #进行归一化
        imRGB[~np.isnan(imRGB)] = (imRGB[~np.isnan(imRGB)] - bottom) / (top - bottom) * num

    if np.ndim(im_orig) == 3:
        outval = np.zeros((h,w,c))
        for i in range(c):
            outval[:,:,i] = np.reshape(imRGB[i,:],(h,w))
    else:
        outval = np.reshape(imRGB,(h,w))

    return outval



def f(path,name,savepath,img2 = None,):

    img = cv2.imread(os.path.join(path,name))
    #img = bi_demo(img)
    #img = cv2.imdecode(np.fromfile(os.path.join(path,name), dtype=np.uint8),-1)
    #print(type(img))
    # s = time.time()
    #img = cv2.resize(img,(1920,1080))
    # t = time.time()
    #print(t - s)
    h, w = img.shape[0], img.shape[1]
    b, g, r = cv2.split(img)
    # global adaptation

    # Y =  0.299 * r + 0.587 * g + 0.114 * b
    # U = -0.147 * r - 0.289 * g + 0.436 *b
    # V = 0.615*r  - 0.515 * g - 0.10 * b
    # r = Y + 1.14  * V
    # g = Y - 0.39 * U - 0.58 * V
    # b = Y + 2.03 * U
    l = 0.299 * r + 0.587 * g + 0.114 * b
    l = l.astype(np.float32)
    #print(type(l[0][0]))
    #print('type ',type(l))
    l_max = np.max(l)
    l_waver = np.exp(np.sum(np.log(0.001 + l)) / (h * w))
    l_log = np.log(l / l_waver + 1) / np.log(l_max / l_waver + 1)
    #print(type(l_log[0][0]))
    # local adaptation
    #kenlRatio = 0.01
    #krnlsz = np.floor(np.max((3, h * kenlRatio, w * kenlRatio)))

    # if krnlsz % 2 == 0 :
    #    krnlsz = krnlsz + 1
    #Lg1 = maxfilter(l_log, int(41))

    Hg = guided_filter2(l_log, l_log, 10, 0.01)

    eta = 36
    alpha = 1 + eta * l_log / np.max(l_log)  # eqn 11   这一块有增强图像效果 看看有没有可以替代的
    # alpha = alpha * (np.power(alpha,(1 / alpha)))
    # bs = np.max(alpha)
    # a = 1.35
    # alpha = 2 * np.arctan(a * alpha / bs) / np.pi * bs

    Lgaver = np.exp(np.sum(np.log(0.001 + l_log)) / (h * w))
    lmda = 3
    beta = lmda * Lgaver
    Lout = alpha * np.log(l_log / Hg + beta)

    #Lout = (Lout - np.min(Lout)) / (np.max(Lout) - np.min(Lout))
    Lout = SimplestColorBalance(Lout,0.005,0.001,1)

    gain = Lout / l

    #gain = l_log / l

    gain[gain == 0] = 0
    out = cv2.merge([gain * b, gain * g, gain * r]) * 255


    out = cv2.convertScaleAbs(out)


    if os.path.isdir(savepath) is not True:
        os.mkdir(savepath)
    cv2.imwrite(os.path.join(savepath,name), out)

def folder_test(path,savepath):
    names = os.listdir(path)
    for i,name in enumerate(names):
        f(path,name,savepath)




def compare(path1,path2,result_path):
    imgs = os.listdir(path1)
    imgs2 = os.listdir(path2)
    if os.path.isdir(result_path) is not True:
        os.mkdir(result_path)

    for i in range(len(imgs)):
        #im1 = cv2.imread(os.path.join(path1,imgs[0]))
        im1 = cv2.imdecode(np.fromfile(os.path.join(path1,imgs[i]), dtype=np.uint8), -1)
        #im2 = cv2.imread(os.path.join(path2, imgs2[0]))
        im2 = cv2.imdecode(np.fromfile(os.path.join(path2, imgs2[i]), dtype=np.uint8), -1)
        result = np.hstack((im1, im2))
        print(os.path.join(result_path, str(i)+'.jpg'))
        cv2.imwrite(os.path.join(result_path, str(i)+'.jpg'),result)


def second_step(save_path,img_path):
    im1 = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
    #im1 = #img#cv2.imread(os.path.join(path,name))
    b , g , r = cv2.split(im1)
    b_tmp = np.zeros((im1.shape[0],im1.shape[1]))
    g_tmp = np.zeros((im1.shape[0],im1.shape[1]))
    r_tmp = np.zeros((im1.shape[0],im1.shape[1]))

    t = (b < 100) & (g < 100) & (r < 100)
    #print('we',)
    b_tmp[t] = b[t]
    g_tmp[t] = g[t]
    r_tmp[t] = r[t]

    result = cv2.merge([b_tmp,g_tmp,r_tmp])

    cv2.imwrite('./cc.png', result)
    result = f('', '', img2=result)
    #result = cv2.imread('./result_second/new.png')

    b2, g2, r2 = cv2.split(result)
    b[t] = b2[t] * 0.25 + b[t] * 0.75
    g[t] = g2[t] * 0.25 + g[t] * 0.75
    r[t] = r2[t] * 0.25 + r[t] * 0.75
    result = cv2.merge([b, g, r])
    cv2.imwrite(save_path, result)

    print('finish')



def mean_shift_demo(image):#均值偏移滤波
    dst = cv2.pyrMeanShiftFiltering(src=image, sp=15, sr=20)
    # cv2.namedWindow('mean_shift image', 0)
    # cv2.resizeWindow('mean_shift image', 300, 400)
    # cv2.imshow("mean_shift image", dst)
    return dst

def bi_demo(image):#高斯双边滤波
    #dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)
    dst = cv2.bilateralFilter(image, 15, 100, 100)
    #cv2.dct()
    cv2.imwrite('./noiss_result.png',dst)
    # cv2.namedWindow('bi_demo',0)
    # cv2.resizeWindow('bi_demo',300,400)
    # cv2.imshow("bi_demo", dst)
    return dst
def single_compare(img1,img2,path):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    print(img1.shape)
    print(img2.shape)
    out = np.hstack((img1,img2))
    cv2.imwrite(path,out)

def fiters():
    ########     四个不同的滤波器    #########
    img = cv2.imread('./img/noiss.jpg')

    # 均值滤波
    img_mean = cv2.blur(img, (5, 5))

    # 高斯滤波
    img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)

    # 中值滤波
    img_median = cv2.medianBlur(img, 5)

    # 双边滤波
    img_bilater = cv2.bilateralFilter(img, 9, 75, 75)

    # 展示不同的图片
    titles = ['srcImg', 'mean', 'Gaussian', 'median', 'bilateral']
    imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

    for i in range(5):
        plt.subplot(2, 3, i + 1)  # 注意，这和matlab中类似，没有0，数组下标从1开始
        plt.imshow(imgs[i])
        plt.title(titles[i])
    plt.show()




if __name__ == '__main__':
    #img = cv2.imread('./img/noiss.jpg')
    #bi_demo(img)
    #single_compare('./img/noiss.jpg','./z63/1603881149143.jpg','./img/noiss_compare.jpg')
    #fiters()
    imgs_path = ''
    save_path = ''
    folder_test(imgs_path,save_path)
    #compare(r'F:\\LowlightDataset\\z6','./z62','./Q2_compare/')

    # img = cv2.imread('./noiss.jpg')
    # d = cv2.getTrackbarPos("d", "image")
    # sigmaColor = cv2.getTrackbarPos("sigmaColor", "image")
    # sigmaSpace = cv2.getTrackbarPos("sigmaSpace", "image")
    # out_img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    # cv2.imwrite('./noiss.jpg',out_img)


